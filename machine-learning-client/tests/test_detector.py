"""Tests for detector.py. The YOLO model is mocked so tests run fast without torch."""

import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

import detector

NAMES = {0: "cup", 1: "book", 2: "laptop"}
WHOLE_IMAGE = (0.0, 0.0, 1.0, 1.0)


def make_image_bytes(image=None, **save_options):
    """Return the bytes of an image, a small blank PNG by default."""
    if image is None:
        image = Image.new("RGB", (20, 20), "white")
    buffer = io.BytesIO()
    image.save(buffer, format=save_options.pop("format", "PNG"), **save_options)
    return buffer.getvalue()


def make_model(detections):
    """Build a fake model that returns the given detections.

    Each detection is (class_id, confidence) or (class_id, confidence, box).
    """
    boxes = []
    for detection in detections:
        cls, conf, box = (*detection, WHOLE_IMAGE)[:3]
        boxes.append(SimpleNamespace(cls=cls, conf=conf, xyxyn=[box]))
    result = SimpleNamespace(names=NAMES, boxes=boxes)
    return MagicMock(return_value=[result])


def make_grid(rows):
    """Turn strings like "#.#" into a grid where # means occupied."""
    return [[char == "#" for char in row] for row in rows]


def square_on_desk():
    """A white 'desk' with a dark square covering the middle quarter."""
    image = Image.new("RGB", (640, 640), "white")
    ImageDraw.Draw(image).rectangle([160, 160, 479, 479], fill="black")
    return image


def test_counts_each_object_type():
    """Objects of the same type are added up."""
    model = make_model([(0, 0.9), (0, 0.8), (1, 0.7), (0, 0.6), (2, 0.95)])
    counts = detector.detect_objects(make_image_bytes(), model=model, threshold=0.5)
    assert counts == {"cup": 3, "book": 1, "laptop": 1}


def test_ignores_low_confidence_detections():
    """Detections under the threshold are not counted."""
    model = make_model([(0, 0.9), (1, 0.3), (2, 0.49)])
    counts = detector.detect_objects(make_image_bytes(), model=model, threshold=0.5)
    assert counts == {"cup": 1}


def test_detection_exactly_at_threshold_is_counted():
    """A confidence equal to the threshold still counts."""
    model = make_model([(1, 0.5)])
    counts = detector.detect_objects(make_image_bytes(), model=model, threshold=0.5)
    assert counts == {"book": 1}


def test_no_objects_returns_empty_dict():
    """An image with nothing detected gives an empty result."""
    model = make_model([])
    counts = detector.detect_objects(make_image_bytes(), model=model)
    assert isinstance(counts, dict)
    assert not counts


def test_model_gets_image_and_threshold():
    """The model is called with a PIL image and the confidence threshold."""
    model = make_model([])
    detector.detect_objects(make_image_bytes(), model=model, threshold=0.25)
    args, kwargs = model.call_args
    assert isinstance(args[0], Image.Image)
    assert kwargs["conf"] == 0.25


def test_uses_default_model_when_none_given():
    """Without a model argument, the shared model from get_model is used."""
    model = make_model([(2, 0.9)])
    with patch.object(detector, "get_model", return_value=model):
        assert detector.detect_objects(make_image_bytes()) == {"laptop": 1}
    model.assert_called_once()


def test_get_model_loads_weights_once():
    """get_model loads the weights file once and caches the model."""
    fake_yolo = MagicMock()
    fake_ultralytics = SimpleNamespace(YOLO=fake_yolo)
    detector.get_model.cache_clear()
    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        first = detector.get_model()
        second = detector.get_model()
    detector.get_model.cache_clear()
    fake_yolo.assert_called_once_with(detector.MODEL_PATH)
    assert first is second


def test_load_image_applies_phone_rotation():
    """A photo tagged as rotated sideways is turned upright."""
    image = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation tag: rotated 90 degrees
    data = make_image_bytes(image, format="JPEG", exif=exif)
    assert detector.load_image(data).size == (20, 40)


def test_find_objects_returns_labels_and_boxes():
    """Each detection comes back with its label and box."""
    model = make_model([(0, 0.9, (0.1, 0.2, 0.3, 0.4)), (1, 0.2)])
    image = Image.new("RGB", (20, 20), "white")
    found = detector.find_objects(image, model=model, threshold=0.5)
    assert found == [("cup", (0.1, 0.2, 0.3, 0.4))]


def test_blank_desk_is_all_negative_space():
    """A plain surface with nothing on it is completely empty."""
    image = Image.new("RGB", (640, 640), "white")
    assert detector.measure_negative_space(image) == 1.0


def test_object_reduces_negative_space():
    """A large object in the middle takes up roughly the space it covers."""
    negative_space = detector.measure_negative_space(square_on_desk())
    assert 0.6 <= negative_space <= 0.8


def test_detected_box_counts_as_occupied():
    """A detected object's box is occupied even if it has no visible edges."""
    image = Image.new("RGB", (640, 640), "white")
    negative_space = detector.measure_negative_space(image, [(0.0, 0.0, 0.5, 0.5)])
    assert negative_space == 0.75


def test_fill_enclosed_fills_inside_of_outline():
    """Cells surrounded by an outline are filled in."""
    grid = make_grid(["#####", "#...#", "#...#", "#####", "....."])
    filled = detector.fill_enclosed(grid)
    assert filled == make_grid(["#####", "#####", "#####", "#####", "....."])


def test_fill_enclosed_leaves_open_shapes_alone():
    """Cells that connect to the edge of the photo stay empty."""
    grid = make_grid(["#####", "#....", "#...#", "#####", "....."])
    assert detector.fill_enclosed(grid) == grid


def test_analyze_desk_returns_objects_and_negative_space():
    """analyze_desk combines the object counts and the negative space."""
    model = make_model([(1, 0.9, (0.0, 0.0, 0.5, 0.5))])
    image = Image.new("RGB", (640, 640), "white")
    result = detector.analyze_desk(make_image_bytes(image), model=model)
    assert result == {"objects": {"book": 1}, "negative_space": 0.75}

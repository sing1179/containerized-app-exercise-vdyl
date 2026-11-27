"""Object detection and negative space measurement for Rate My Desk."""

import io
import os
from collections import Counter
from functools import lru_cache

from PIL import Image, ImageFilter, ImageOps

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8s.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))

# negative space settings
GRID_SIZE = 16  # the photo is split into a 16 x 16 grid of cells
ANALYSIS_WIDTH = 640  # photos are shrunk to this size before looking for edges
EDGE_BLUR = 1.5  # blur first so desk textures like wood grain don't count as clutter
EDGE_PIXEL_THRESHOLD = 25  # how strong an edge has to be to count
BUSY_CELL_THRESHOLD = 8  # how much edge a cell needs before it counts as occupied


@lru_cache(maxsize=1)
def get_model():
    """Load the YOLO model once and reuse it for every scan."""
    # imported here so the worker and tests don't load torch until needed
    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

    return YOLO(MODEL_PATH)


def load_image(image_bytes):
    """Open image bytes as an upright RGB image."""
    image = Image.open(io.BytesIO(image_bytes))
    # phone photos are often stored sideways with a rotation tag
    return ImageOps.exif_transpose(image).convert("RGB")


def find_objects(image, model=None, threshold=CONFIDENCE_THRESHOLD):
    """Return a (label, box) pair for each object found in the image.

    Each box is (x1, y1, x2, y2) as fractions of the image width and height.
    """
    if model is None:
        model = get_model()

    detections = []
    for result in model(image, conf=threshold, verbose=False):
        for box in result.boxes:
            if float(box.conf) >= threshold:
                label = result.names[int(box.cls)]
                coords = tuple(float(value) for value in box.xyxyn[0])
                detections.append((label, coords))
    return detections


def detect_objects(image_bytes, model=None, threshold=CONFIDENCE_THRESHOLD):
    """Run detection on an image and return a count of each object type.

    Example return value: {"cup": 3, "book": 5}
    """
    detections = find_objects(load_image(image_bytes), model, threshold)
    return dict(Counter(label for label, _ in detections))


def edge_grid(image, grid_size=GRID_SIZE):
    """Mark each grid cell True if it has enough edges to look occupied."""
    gray = image.convert("L")
    gray.thumbnail((ANALYSIS_WIDTH, ANALYSIS_WIDTH))
    gray = ImageOps.autocontrast(gray, cutoff=1)
    edges = gray.filter(ImageFilter.GaussianBlur(EDGE_BLUR))
    edges = edges.filter(ImageFilter.FIND_EDGES)

    # the edge filter always lights up the photo's border, so trim it off
    width, height = edges.size
    edges = edges.crop((2, 2, width - 2, height - 2))

    strong = edges.point(lambda value: 255 if value > EDGE_PIXEL_THRESHOLD else 0)
    # shrinking to the grid size averages each cell into a single pixel
    cells = strong.resize((grid_size, grid_size), Image.Resampling.BOX).tobytes()
    return [
        [cells[row * grid_size + col] > BUSY_CELL_THRESHOLD for col in range(grid_size)]
        for row in range(grid_size)
    ]


def mark_boxes(grid, boxes):
    """Mark every cell whose center falls inside a detected object's box."""
    size = len(grid)
    for row in range(size):
        for col in range(size):
            x, y = (col + 0.5) / size, (row + 0.5) / size
            if any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in boxes):
                grid[row][col] = True
    return grid


def fill_enclosed(grid):
    """Mark free cells that are fully surrounded by occupied cells.

    Edges only find the outline of an object, so the inside of something
    like a notebook would otherwise count as empty desk.
    """
    size = len(grid)
    reachable = set()
    to_visit = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if row in (0, size - 1) or col in (0, size - 1)
    ]
    while to_visit:
        row, col = to_visit.pop()
        if (row, col) in reachable or not 0 <= row < size or not 0 <= col < size:
            continue
        if grid[row][col]:
            continue
        reachable.add((row, col))
        to_visit += [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]

    return [
        [grid[row][col] or (row, col) not in reachable for col in range(size)]
        for row in range(size)
    ]


def measure_negative_space(image, boxes=()):
    """Return the fraction of the desk that is empty, from 0.0 to 1.0.

    A cell counts as occupied if it has strong edges, sits inside a detected
    object, or is enclosed by occupied cells.
    """
    grid = fill_enclosed(mark_boxes(edge_grid(image), boxes))
    occupied = sum(cell for row in grid for cell in row)
    return round(1 - occupied / (len(grid) ** 2), 2)


def analyze_desk(image_bytes, model=None, threshold=CONFIDENCE_THRESHOLD):
    """Detect objects and measure negative space in one pass.

    Example return value: {"objects": {"cup": 1, "book": 1}, "negative_space": 0.72}
    """
    image = load_image(image_bytes)
    detections = find_objects(image, model, threshold)
    return {
        "objects": dict(Counter(label for label, _ in detections)),
        "negative_space": measure_negative_space(image, [box for _, box in detections]),
    }

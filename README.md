![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Containerized App Exercise
# Rate My Desk

Take a photo of your desk and find out how messy it really is. The machine learning client spots the objects on your desk with YOLOv8 and turns the clutter into a score and a tier, from spotless to disaster.

The web app lets you sign up, snap or upload a photo, see your result, look back at your history, and compare desks on the leaderboard.

## Team

- sing1179
- YusefMoustafa
- denisekos
- 9021007

## How it works

The system runs as three Docker containers:

- **mongodb** - the database, run from the official `mongo` image
- **web-app** - a Flask app in [`web-app/`](./web-app) where users log in, upload photos and see results
- **machine-learning-client** - a Python worker in [`machine-learning-client/`](./machine-learning-client) that picks up new photos, detects objects and saves the score

The web app saves each photo as a pending scan in MongoDB. The machine learning client finds pending scans, analyzes them and writes the results back, and the web app shows them.

## Configuration

The app reads its settings from a `.env` file in the project root. Copy the example and edit it:

```bash
cp env.example .env
```

On Windows, use `copy env.example .env`.

| Variable | What it does |
| --- | --- |
| `MONGO_URI` | MongoDB connection string. Use `mongodb://mongodb:27017` with Docker, or `mongodb://localhost:27017` when running the web app directly. |
| `MONGO_DBNAME` | Database name. Defaults to `rate_my_desk`. |
| `SECRET_KEY` | Key Flask uses to sign login sessions. Set it to any long random string. |

No starter data is needed. Users and scans are created as you use the app.

## Running with Docker Compose

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
docker compose up --build
```

This currently starts MongoDB and the web app. Open [http://localhost:8080](http://localhost:8080). To stop everything, press `Ctrl+C` and run:

```bash
docker compose down
```

If port 27017 is already in use, stop any other MongoDB container first with `docker stop mongodb`.

## Running the web app for development

You need Python 3.13 and [pipenv](https://pipenv.pypa.io/). In your `.env`, set `MONGO_URI=mongodb://localhost:27017`.

Start MongoDB. The first time, run:

```bash
docker run --name mongodb -d -p 27017:27017 mongo
```

After that, use `docker start mongodb`.

Then run the web app:

```bash
cd web-app
pipenv install --dev
pipenv run python app.py
```

The app runs on [http://localhost:5000](http://localhost:5000). On macOS, AirPlay can hold port 5000; if you see "Address already in use", run `PORT=5001 pipenv run python app.py` instead.

## Tests and linting

From `web-app/`:

```bash
pipenv run python -m pytest --cov=. --cov-fail-under=80
pipenv run pylint --recursive=y .
pipenv run black --check .
```
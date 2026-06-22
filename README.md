# Face Recognition System

A realtime face recognition project based on the pipeline described in the course document:

```text
Camera/Image -> SCRFD Face Detection -> Face Alignment -> ArcFace Embedding -> FAISS Matching -> Identity
```

This version does **not** require training. It uses pretrained InsightFace ONNX models from the `buffalo_l` model pack:

- `det_10g.onnx`: SCRFD face detector, returns bounding boxes and 5 facial landmarks.
- `w600k_r50.onnx`: ArcFace recognizer, returns a 512-dimensional face embedding.

The system supports enrolling known faces, storing embeddings in a FAISS vector database, recognizing faces from images, and running realtime recognition from a webcam.

## Features

- Face detection with SCRFD pretrained model
- Face alignment from 5 landmarks
- Face embedding extraction with ArcFace
- Cosine similarity search with FAISS
- FastAPI backend with Swagger UI
- Image enrollment API
- Image recognition API
- Realtime webcam client with OpenCV
- One-line Windows scripts for API and camera

## Project Structure

```text
code/
  backend/
    app.py              # FastAPI application
    face_engine.py      # SCRFD + ArcFace + FAISS pipeline
    data/               # Generated FAISS index and enrolled persons metadata
    models/             # Downloaded InsightFace ONNX models
  client/
    camera_client.py    # Realtime webcam client
  scripts/
    enroll_image.py     # Enroll one image
    enroll_folder.py    # Enroll many images grouped by person name
    recognize_image.py  # Recognize faces in one image
  api.bat               # Start backend only
  camera.bat            # Start camera client only
  run.bat               # Start backend and camera
  requirements.txt
  README.md
```

## Requirements

Recommended Python version:

```text
Python 3.10, 3.11, or 3.12
```

Python 3.13/3.14 may not work well because some ML packages may not provide compatible wheels yet.

## Installation

Clone the repository and open the project folder:

```powershell
cd "D:\AIO\Project\Face Recognition System\code"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The first run requires internet access because the backend downloads the pretrained InsightFace `buffalo_l` model pack automatically.

Downloaded models are stored in:

```text
backend/models/
```

## Run The API

Start the backend:

```powershell
.\api.bat
```

Or manually:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/
```

## Enroll A Face

Before recognition, enroll at least one known face.

Using Swagger UI:

1. Open `http://127.0.0.1:8000/docs`.
2. Open `POST /enroll`.
3. Enter a name.
4. Upload a clear face image.
5. Execute the request.

Using command line:

```powershell
.\.venv\Scripts\python.exe scripts\enroll_image.py --name "Tien" --image "D:\path\to\tien.jpg"
```

Enroll multiple people from folders:

```text
known_faces/
  Tien/
    1.jpg
    2.jpg
  Nha/
    1.jpg
    2.jpg
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\enroll_folder.py --folder "D:\path\to\known_faces"
```

The face database is saved in:

```text
backend/data/faces.index
backend/data/persons.json
```

## Recognize One Image

Using command line:

```powershell
.\.venv\Scripts\python.exe scripts\recognize_image.py --image "D:\path\to\test.jpg"
```

Using Swagger UI:

- `POST /recognize`: returns JSON results.
- `POST /recognize-image`: returns an image with bounding boxes and names drawn on it.

Example JSON response:

```json
{
  "faces": [
    {
      "name": "Tien",
      "person_id": 0,
      "score": 0.72,
      "bbox": [120, 80, 260, 240],
      "landmarks": [[155.2, 130.4], [220.1, 128.9], [188.5, 165.7], [162.3, 205.1], [215.8, 203.6]]
    }
  ],
  "database_size": 1
}
```

## Run Realtime Camera

Make sure the API is running first, then start the camera client:

```powershell
.\camera.bat
```

Or manually:

```powershell
.\.venv\Scripts\python.exe client\camera_client.py
```

Press `q` in the camera window to quit.

## One-Line Demo

To start the backend and camera together:

```powershell
.\run.bat
```

Note: enroll known faces before using the realtime demo, otherwise all detected faces will be labeled as `unknown`.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Health check and database size |
| `GET` | `/persons` | List enrolled face records |
| `POST` | `/enroll` | Enroll a face image with a name |
| `POST` | `/recognize` | Recognize faces and return JSON |
| `POST` | `/recognize-image` | Return an annotated image |
| `DELETE` | `/database` | Reset FAISS index and person metadata |

## Recognition Threshold

The default recognition threshold is set in `backend/app.py`:

```python
engine = FaceEngine(threshold=0.45)
```

- Increase the threshold to make recognition stricter.
- Decrease the threshold to make recognition more permissive.

Suggested range:

```text
0.40 - 0.60
```

## How It Works

1. SCRFD detects faces and returns bounding boxes plus 5 landmarks.
2. The landmarks are used to align each face to the ArcFace 112x112 template.
3. ArcFace extracts a 512-dimensional embedding.
4. The embedding is L2-normalized.
5. FAISS searches the nearest enrolled embedding using inner product, equivalent to cosine similarity after normalization.
6. If the similarity score is above the threshold, the matching identity is returned. Otherwise, the face is labeled as `unknown`.

## Notes For GitHub

Generated runtime files should not be committed:

```text
.venv/
backend/models/
backend/data/
__pycache__/
```

Recommended `.gitignore`:

```gitignore
.venv/
__pycache__/
*.pyc
backend/models/
backend/data/
```

## Credits

- Face detection and recognition models: InsightFace
- Vector similarity search: FAISS
- Backend framework: FastAPI
- Camera processing: OpenCV

## Web UI

The project now includes a browser-based interface with two main functions:

1. Enroll a face by entering a name and uploading/capturing a face image.
2. Recognize faces from the camera in realtime and draw bounding boxes, landmarks, names, and similarity scores.

Start the app:

```powershell
.\run.bat
```

Then open:

```text
http://127.0.0.1:8000/
```

The API documentation is still available at:

```text
http://127.0.0.1:8000/docs
```

The health endpoint is:

```text
http://127.0.0.1:8000/health
```

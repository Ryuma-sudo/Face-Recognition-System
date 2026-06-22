from pathlib import Path
from typing import Annotated

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

try:
    from .face_engine import FaceEngine
except ImportError:
    from face_engine import FaceEngine


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Face Recognition System")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
engine = FaceEngine(threshold=0.45)


async def _read_upload(file: UploadFile):
    image_bytes = await file.read()
    return FaceEngine.read_image(image_bytes)


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "database_size": engine.index.ntotal,
        "threshold": engine.threshold,
    }


@app.get("/persons")
def list_persons():
    return {
        "count": len(engine.persons),
        "persons": engine.persons,
    }


@app.post("/enroll")
async def enroll(
    name: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
):
    try:
        image = await _read_upload(file)
        return engine.enroll(image, name=name, source=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/recognize")
async def recognize(file: Annotated[UploadFile, File()]):
    try:
        image = await _read_upload(file)
        return {
            "faces": engine.recognize(image),
            "database_size": engine.index.ntotal,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/recognize-image")
async def recognize_image(file: Annotated[UploadFile, File()]):
    try:
        image = await _read_upload(file)
        results = engine.recognize(image)
        output = engine.draw_results(image, results)
        ok, buffer = cv2.imencode(".jpg", output)
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode output image.")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/database")
def reset_database():
    engine.reset_database()
    return {"message": "database reset"}

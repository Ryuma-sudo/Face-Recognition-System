import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import cv2
import faiss
import numpy as np
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
INDEX_PATH = DATA_DIR / "faces.index"
PERSONS_PATH = DATA_DIR / "persons.json"
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DET_MODEL_PATH = MODEL_DIR / "det_10g.onnx"
REC_MODEL_PATH = MODEL_DIR / "w600k_r50.onnx"
ARCFACE_DST = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceEngine:
    def __init__(
        self,
        threshold: float = 0.45,
        duplicate_threshold: float = 0.80,
        det_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
    ) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        self.threshold = threshold
        self.duplicate_threshold = duplicate_threshold
        self.embedding_dim = 512
        self.providers = providers or ["CPUExecutionProvider"]
        self.det_size = det_size
        self._ensure_models()

        self.det_session = ort.InferenceSession(str(DET_MODEL_PATH), providers=self.providers)
        self.rec_session = ort.InferenceSession(str(REC_MODEL_PATH), providers=self.providers)
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.rec_input_name = self.rec_session.get_inputs()[0].name

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.persons: list[dict[str, Any]] = []
        self.load_database()

    def _ensure_models(self) -> None:
        if DET_MODEL_PATH.exists() and REC_MODEL_PATH.exists():
            return

        zip_path = MODEL_DIR / "buffalo_l.zip"
        try:
            print("Downloading InsightFace buffalo_l model pack...")
            urllib.request.urlretrieve(MODEL_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(MODEL_DIR)
        except Exception as exc:
            raise RuntimeError(
                "Could not download buffalo_l models. Check internet connection or "
                f"manually download {MODEL_URL} and extract it to {MODEL_DIR}."
            ) from exc

        if not DET_MODEL_PATH.exists() or not REC_MODEL_PATH.exists():
            raise RuntimeError(
                "buffalo_l model pack is missing det_10g.onnx or w600k_r50.onnx."
            )

    def load_database(self) -> None:
        if INDEX_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))

        if PERSONS_PATH.exists():
            with PERSONS_PATH.open("r", encoding="utf-8") as f:
                self.persons = json.load(f)

        if self.index.ntotal != len(self.persons):
            raise RuntimeError(
                "Database is inconsistent: FAISS index size does not match persons.json"
            )

    def save_database(self) -> None:
        faiss.write_index(self.index, str(INDEX_PATH))
        with PERSONS_PATH.open("w", encoding="utf-8") as f:
            json.dump(self.persons, f, ensure_ascii=False, indent=2)

    def reset_database(self) -> None:
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.persons = []
        self.save_database()

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        input_blob, det_scale = self._preprocess_detection(image)
        outputs = self.det_session.run(None, {self.det_input_name: input_blob})
        faces = self._postprocess_detection(outputs, det_scale)
        return sorted(faces, key=lambda face: self._bbox_area(face["bbox"]), reverse=True)

    def enroll(self, image: np.ndarray, name: str, source: str | None = None) -> dict[str, Any]:
        faces = self.detect(image)
        if not faces:
            raise ValueError("No face found in the image.")

        face = faces[0]
        embedding = self._get_embedding(image, face["landmarks"])
        duplicate = self._find_duplicate(embedding)
        if duplicate is not None:
            raise ValueError(
                f"Face already enrolled as {duplicate['name']} "
                f"(similarity {duplicate['score']:.2f})."
            )

        self.index.add(np.asarray([embedding], dtype=np.float32))

        record = {
            "id": len(self.persons),
            "name": name.strip(),
            "source": source or "",
            "bbox": self._bbox_to_list(face["bbox"]),
        }
        self.persons.append(record)
        self.save_database()

        return {
            "message": "enrolled",
            "person": record,
            "faces_detected": len(faces),
        }

    def _find_duplicate(self, embedding: np.ndarray) -> dict[str, Any] | None:
        if self.index.ntotal == 0:
            return None

        scores, indices = self.index.search(
            np.asarray([embedding], dtype=np.float32),
            k=1,
        )
        score = float(scores[0][0])
        match_idx = int(indices[0][0])

        if score < self.duplicate_threshold or not 0 <= match_idx < len(self.persons):
            return None

        return {
            "name": self.persons[match_idx]["name"],
            "person_id": self.persons[match_idx]["id"],
            "score": score,
        }

    def recognize(self, image: np.ndarray) -> list[dict[str, Any]]:
        faces = self.detect(image)
        results: list[dict[str, Any]] = []

        for face in faces:
            embedding = self._get_embedding(image, face["landmarks"])
            name = "unknown"
            score = 0.0
            person_id = None

            if self.index.ntotal > 0:
                scores, indices = self.index.search(
                    np.asarray([embedding], dtype=np.float32),
                    k=1,
                )
                score = float(scores[0][0])
                match_idx = int(indices[0][0])

                if score >= self.threshold and 0 <= match_idx < len(self.persons):
                    match = self.persons[match_idx]
                    name = match["name"]
                    person_id = match["id"]

            results.append(
                {
                    "name": name,
                    "person_id": person_id,
                    "score": score,
                    "bbox": self._bbox_to_list(face["bbox"]),
                    "landmarks": np.asarray(face["landmarks"]).round(2).tolist(),
                }
            )

        return results

    def draw_results(self, image: np.ndarray, results: list[dict[str, Any]]) -> np.ndarray:
        output = image.copy()

        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            known = result["name"] != "unknown"
            color = (0, 200, 0) if known else (0, 0, 255)
            label = f'{result["name"]} {result["score"]:.2f}'

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

            for x, y in result.get("landmarks", []):
                cv2.circle(output, (int(x), int(y)), 2, (255, 255, 0), -1)

        return output

    @staticmethod
    def read_image(image_bytes: bytes) -> np.ndarray:
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image file.")
        return image

    def _preprocess_detection(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        input_w, input_h = self.det_size
        img_h, img_w = image.shape[:2]
        scale = min(input_w / img_w, input_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=(input_w, input_h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob.astype(np.float32), scale

    def _postprocess_detection(
        self,
        outputs: list[np.ndarray],
        det_scale: float,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> list[dict[str, Any]]:
        input_w, input_h = self.det_size
        strides = [8, 16, 32]
        scores_list = outputs[:3]
        bbox_list = outputs[3:6]
        kps_list = outputs[6:9]

        all_boxes = []
        all_landmarks = []
        all_scores = []

        for stride, scores, bbox_preds, kps_preds in zip(
            strides, scores_list, bbox_list, kps_list
        ):
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            kps_preds = kps_preds.reshape(-1, 10)

            height = input_h // stride
            width = input_w // stride
            anchor_count = max(1, scores.shape[0] // (height * width))
            centers = self._anchor_centers(width, height, stride, anchor_count)

            keep = np.where(scores >= score_threshold)[0]
            if keep.size == 0:
                continue

            centers_keep = centers[keep]
            bbox_keep = bbox_preds[keep]
            kps_keep = kps_preds[keep]

            boxes = np.column_stack(
                [
                    centers_keep[:, 0] - bbox_keep[:, 0] * stride,
                    centers_keep[:, 1] - bbox_keep[:, 1] * stride,
                    centers_keep[:, 0] + bbox_keep[:, 2] * stride,
                    centers_keep[:, 1] + bbox_keep[:, 3] * stride,
                ]
            )

            landmarks = np.zeros((keep.size, 5, 2), dtype=np.float32)
            for j in range(5):
                landmarks[:, j, 0] = centers_keep[:, 0] + kps_keep[:, j * 2] * stride
                landmarks[:, j, 1] = centers_keep[:, 1] + kps_keep[:, j * 2 + 1] * stride

            all_boxes.append(boxes / det_scale)
            all_landmarks.append(landmarks / det_scale)
            all_scores.append(scores[keep])

        if not all_boxes:
            return []

        boxes = np.vstack(all_boxes).astype(np.float32)
        landmarks = np.vstack(all_landmarks).astype(np.float32)
        scores = np.concatenate(all_scores).astype(np.float32)
        keep = self._nms(boxes, scores, nms_threshold)

        return [
            {
                "bbox": boxes[i],
                "landmarks": landmarks[i],
                "score": float(scores[i]),
            }
            for i in keep
        ]

    @staticmethod
    def _anchor_centers(width: int, height: int, stride: int, anchor_count: int) -> np.ndarray:
        y, x = np.mgrid[:height, :width]
        centers = np.stack((x, y), axis=-1).astype(np.float32)
        centers = (centers.reshape(-1, 2) + 0.5) * stride
        if anchor_count > 1:
            centers = np.repeat(centers, anchor_count, axis=0)
        return centers

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            overlap = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(overlap <= threshold)[0]
            order = order[inds + 1]

        return keep

    def _get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        aligned = self._align_face(image, landmarks)
        blob = cv2.dnn.blobFromImage(
            aligned,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        ).astype(np.float32)
        embedding = self.rec_session.run(None, {self.rec_input_name: blob})[0][0]
        return self._normalize(embedding)

    @staticmethod
    def _align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        src = np.asarray(landmarks, dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST, method=cv2.LMEDS)
        if matrix is None:
            raise ValueError("Could not align face from landmarks.")
        return cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Face embedding has zero norm.")
        return embedding / norm

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    @staticmethod
    def _bbox_to_list(bbox: np.ndarray) -> list[int]:
        return [int(v) for v in np.asarray(bbox).round().tolist()]

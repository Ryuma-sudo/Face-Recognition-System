import argparse
import time

import cv2
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime face recognition camera client.")
    parser.add_argument("--api", default="http://127.0.0.1:8000/recognize")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=960)
    return parser.parse_args()


def resize_frame(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


def recognize_frame(api_url, frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return []

    response = requests.post(
        api_url,
        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
        timeout=5,
    )
    response.raise_for_status()
    return response.json().get("faces", [])


def draw(frame, faces):
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        name = face["name"]
        score = face["score"]
        known = name != "unknown"
        color = (0, 220, 0) if known else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {score:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        for x, y in face.get("landmarks", []):
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1)


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    faces = []
    last_request = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, args.width)
        now = time.time()

        if now - last_request >= args.interval:
            last_request = now
            try:
                faces = recognize_frame(args.api, frame)
            except requests.RequestException:
                cv2.putText(
                    frame,
                    "API offline",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        draw(frame, faces)
        cv2.imshow("Face Recognition System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

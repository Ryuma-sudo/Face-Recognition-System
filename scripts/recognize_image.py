import argparse
from pathlib import Path

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Recognize faces in one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000/recognize")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    with image_path.open("rb") as f:
        response = requests.post(
            args.api,
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )

    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()

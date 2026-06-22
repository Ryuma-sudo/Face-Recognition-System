import argparse
from pathlib import Path

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Enroll one face image through the API.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000/enroll")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    with image_path.open("rb") as f:
        response = requests.post(
            args.api,
            data={"name": args.name},
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )

    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()

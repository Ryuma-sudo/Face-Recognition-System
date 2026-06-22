import argparse
from pathlib import Path

import requests


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enroll images organized as known_faces/person_name/image.jpg."
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000/enroll")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.folder)

    if not root.exists():
        raise FileNotFoundError(root)

    for person_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        name = person_dir.name
        for image_path in sorted(person_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            with image_path.open("rb") as f:
                response = requests.post(
                    args.api,
                    data={"name": name},
                    files={"file": (image_path.name, f, "image/jpeg")},
                    timeout=30,
                )

            if response.ok:
                print(f"OK   {name}: {image_path}")
            else:
                print(f"FAIL {name}: {image_path} -> {response.text}")


if __name__ == "__main__":
    main()

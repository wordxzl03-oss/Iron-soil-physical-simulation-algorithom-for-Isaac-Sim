"""Create a contact sheet from representative frames of a GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gif", type=Path)
    parser.add_argument("--output", type=Path, default=Path("contact_sheet.png"))
    args = parser.parse_args()
    image = Image.open(args.gif)
    total = image.n_frames
    indices = [round(value) for value in [0, total * 0.28, total * 0.52, total * 0.72, total - 1]]
    frames = []
    for index in indices:
        image.seek(min(index, total - 1))
        frame = image.convert("RGB")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((8, 8, 112, 34), fill="white")
        draw.text((14, 13), f"frame {index}", fill="black")
        frames.append(frame)
    sheet = Image.new("RGB", (frames[0].width * len(frames), frames[0].height), "white")
    for column, frame in enumerate(frames):
        sheet.paste(frame, (column * frame.width, 0))
    sheet.save(args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

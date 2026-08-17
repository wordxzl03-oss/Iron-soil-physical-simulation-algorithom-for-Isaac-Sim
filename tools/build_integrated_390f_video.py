"""Build a compact, state-changing MP4 from an Isaac Replicator frame stream.

Replicator may emit many nearly identical frames while the reduced-order soil
solver is busy.  This utility keeps frames only after a measurable image-state
change, caps the result deterministically, and preserves the selected PNGs as
auditable video source frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def _frame_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _thumbnail(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize((160, 90), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )


def _uniform_cap(paths: list[Path], maximum: int) -> list[Path]:
    if len(paths) <= maximum:
        return paths
    indices = np.linspace(0, len(paths) - 1, maximum, dtype=np.int64)
    return [paths[index] for index in np.unique(indices)]


def select_frames(
    frame_dir: Path,
    *,
    maximum_source_index: int,
    mean_difference_threshold: float,
    maximum_output_frames: int,
) -> tuple[list[Path], dict[str, float | int]]:
    candidates = sorted(frame_dir.glob("rgb_*.png"), key=_frame_index)
    candidates = [
        path for path in candidates if _frame_index(path) <= maximum_source_index
    ]
    if not candidates:
        raise RuntimeError(f"no source frames found in {frame_dir}")

    selected = [candidates[0]]
    previous = _thumbnail(candidates[0])
    for path in candidates[1:-1]:
        current = _thumbnail(path)
        difference = float(np.mean(np.abs(current - previous)))
        if difference >= mean_difference_threshold:
            selected.append(path)
        previous = current
    if candidates[-1] != selected[-1]:
        selected.append(candidates[-1])
    selected = _uniform_cap(selected, maximum_output_frames)
    return selected, {
        "source_frame_count": len(candidates),
        "selected_frame_count": len(selected),
        "maximum_source_index": maximum_source_index,
        "mean_difference_threshold": mean_difference_threshold,
        "maximum_output_frames": maximum_output_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-dir", type=Path, required=True)
    parser.add_argument("--maximum-source-index", type=int, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--mean-difference-threshold", type=float, default=0.25)
    parser.add_argument("--maximum-output-frames", type=int, default=240)
    args = parser.parse_args()

    selected, manifest = select_frames(
        args.frame_dir,
        maximum_source_index=args.maximum_source_index,
        mean_difference_threshold=args.mean_difference_threshold,
        maximum_output_frames=args.maximum_output_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.selected_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in args.selected_dir.glob("frame_*.png"):
        stale_frame.unlink()

    copied: list[Path] = []
    for index, source in enumerate(selected):
        destination = args.selected_dir / f"frame_{index:04d}.png"
        shutil.copy2(source, destination)
        copied.append(destination)

    with imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for path in copied:
            writer.append_data(imageio.imread(path))

    manifest.update(
        {
            "fps": args.fps,
            "duration_s": len(copied) / args.fps,
            "video": str(args.output),
            "selected_frame_directory": str(args.selected_dir),
            "first_source_frame": selected[0].name,
            "last_source_frame": selected[-1].name,
        }
    )
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

"""Build a simple rectangular wheel loader OBJ with wheels, boom and bucket."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from render_obj_loader_scooping import (
    _prepare_model,
    infer_landmarks,
    load_obj_parts,
)


def box_mesh(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    vertices = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return vertices, faces


def wheel_mesh(
    center: tuple[float, float, float],
    width: float = 0.46,
    radius: float = 0.72,
    segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Cylinder whose axle is parallel to world X."""
    cx, cy, cz = center
    vertices = []
    for x in (cx - width / 2, cx + width / 2):
        for angle in np.linspace(0, 2 * np.pi, segments, endpoint=False):
            vertices.append([x, cy + radius * np.cos(angle), cz + radius * np.sin(angle)])
    vertices.extend([[cx - width / 2, cy, cz], [cx + width / 2, cy, cz]])
    faces = []
    left_center, right_center = 2 * segments, 2 * segments + 1
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend([
            [index, segments + index, segments + nxt],
            [index, segments + nxt, nxt],
            [left_center, nxt, index],
            [right_center, segments + index, segments + nxt],
        ])
    return np.asarray(vertices), np.asarray(faces)


def write_obj(
    path: Path,
    objects: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    lines = [
        "# Simple wheel loader generated from wheel_buck.obj",
        "# +Y is forward; Z is up; units are metres",
    ]
    vertex_offset = 1
    for name, vertices, faces in objects:
        lines.append(f"o {name}")
        lines.extend(
            f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}"
            for vertex in vertices
        )
        lines.extend(
            "f " + " ".join(str(int(index) + vertex_offset) for index in face)
            for face in faces
        )
        vertex_offset += len(vertices)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(source: Path, output: Path) -> None:
    raw = load_obj_parts(source)
    parts, _ = _prepare_model(raw, infer_landmarks(raw))
    chassis = box_mesh((-1.30, -4.20, 0.68), (1.30, -0.35, 2.25))
    cab = box_mesh((-1.12, -3.60, 2.25), (1.12, -1.65, 3.45))
    objects: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("simple_chassis", *chassis),
        ("simple_cab", *cab),
    ]
    for name, x, y in (
        ("wheel_front_left", -1.42, -1.05),
        ("wheel_front_right", 1.42, -1.05),
        ("wheel_rear_left", -1.42, -3.35),
        ("wheel_rear_right", 1.42, -3.35),
    ):
        objects.append((name, *wheel_mesh((x, y, 0.74))))
    boom = parts["loader_boom_world.stl"]
    bucket = parts["loader_bucket_world.stl"]
    objects.extend([
        ("loader_boom", boom.vertices, boom.faces),
        ("loader_bucket", bucket.vertices, bucket.faces),
    ])
    write_obj(output, objects)
    print(f"saved: {output}")
    print("objects:", ", ".join(name for name, _, _ in objects))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("wheel_buck.obj"))
    parser.add_argument(
        "--output", type=Path, default=Path("simple_wheel_loader.obj")
    )
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()

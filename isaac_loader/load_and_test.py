"""Load wheel_loader.usd in Isaac Sim and exercise steering/lift/bucket DOFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usd", type=Path, default=Path(__file__).with_name("wheel_loader.usd")
    )
    args = parser.parse_args()
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(str(args.usd.resolve()), "/World/WheelLoader")
    loader = world.scene.add(
        Articulation("/World/WheelLoader/rear_chassis", name="wheel_loader")
    )
    world.reset()
    print("DOFs:", loader.dof_names)

    controller = loader.get_articulation_controller()
    dof_names = loader.dof_names
    targets = np.zeros(len(dof_names), dtype=np.float32)
    commands = {
        "articulation_joint": np.deg2rad(12),
        "lift_joint": np.deg2rad(18),
        "bucket_joint": np.deg2rad(28),
    }
    for name, value in commands.items():
        if name not in dof_names:
            raise RuntimeError(f"missing expected DOF: {name}; got {dof_names}")
        targets[dof_names.index(name)] = value

    for _ in range(600):
        controller.apply_action(
            ArticulationAction(joint_positions=targets)
        )
        world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()

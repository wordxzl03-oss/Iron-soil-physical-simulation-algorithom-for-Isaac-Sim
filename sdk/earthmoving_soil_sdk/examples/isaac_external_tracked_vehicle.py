"""Attach soil to an already-created, independently controlled Isaac vehicle.

This file intentionally does not create SimulationApp, World, an articulation,
or a controller. Supply objects from your existing Isaac application.
"""

from earthmoving_soil import SoilPhysics, ToolGeometry, TrackGeometry
from earthmoving_soil.isaac import IsaacSoilAdapter


def attach_soil_to_existing_vehicle(
    *, stage, soil_config, initial_heightmap_m,
    bucket_rigid_prim, left_track_rigid_prim, right_track_rigid_prim,
):
    soil = SoilPhysics(soil_config, initial_heightmap_m)
    soil.register_tool("bucket", ToolGeometry.parameterized_bucket(
        width_m=2.0, mouth_depth_m=1.1, rear_height_m=0.9, capacity_m3=1.5,
    ))
    soil.register_track("left_track", TrackGeometry(4.5, 0.65))
    soil.register_track("right_track", TrackGeometry(4.5, 0.65))

    adapter = IsaacSoilAdapter(
        stage, soil, sync_visual_mesh=True,
        sync_contact_surface=False, apply_reaction_wrench=True,
    )
    adapter.bind_tool_prim("bucket", bucket_rigid_prim)
    adapter.bind_track_prim("left_track", left_track_rigid_prim)
    adapter.bind_track_prim("right_track", right_track_rigid_prim)
    adapter.initialize(timestamp_s=0.0)
    return adapter


def soil_step_inside_your_existing_loop(adapter, sim_time_s, physics_dt_s,
                                        left_belt_speed, right_belt_speed):
    # Your controller has already commanded/moved the vehicle. Soil only
    # observes bodies, returns/applies reaction, and publishes deforming H_free.
    result = adapter.step(
        physics_dt_s,
        timestamp_s=sim_time_s,
        phase="coordinated_cut",
        belt_speeds={
            "left_track": left_belt_speed,
            "right_track": right_belt_speed,
        },
    )
    return result.rl_feedback


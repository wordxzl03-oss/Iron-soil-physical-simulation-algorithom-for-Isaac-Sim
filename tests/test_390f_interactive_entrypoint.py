from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_one_command_gui_entrypoint_exists_and_defaults_are_interactive():
    launcher = ROOT / "run_390f_interactive.sh"
    source = launcher.read_text(encoding="utf-8")
    assert launcher.stat().st_mode & 0o111
    assert "isaac_loader/run_390f_v2.py" in source
    config = yaml.safe_load((ROOT / "configs/390f_v2_interactive.yaml").read_text(encoding="utf-8"))
    assert config["runtime"]["headless"] is False
    assert config["runtime"]["wait_for_user"] is True
    assert config["visualization"]["record_video"] is False


def test_runtime_has_required_controls_and_no_motion_teleport():
    source = (ROOT / "isaac_loader/run_390f_v2.py").read_text(encoding="utf-8")
    for label in (
        "START / 1 CYCLE", "RUN 3 CYCLES", "PAUSE", "RESUME",
        "EMERGENCY STOP", "RESET ALL", "NO SOIL", "QUASI STATIC",
        "FULL SOIL", "TRACK SOIL ON/OFF", "KINEMATIC DRIVE DEBUG",
        "Swing", "Boom", "Stick", "Bucket", "Left Track", "Right Track",
    ):
        assert label in source
    assert ".set_world_pose(" not in source
    assert ".set_local_pose(" not in source
    assert "autoplay\": False" in source
    assert "root_pose_write_count\": 0" in source


def test_full_physics_is_blocked_by_failed_no_soil_gate_and_failure_stays_open():
    source = (ROOT / "isaac_loader/run_390f_v2.py").read_text(encoding="utf-8")
    assert "NO_SOIL_MACHINE_GATE_BLOCKED" in source
    assert "Isaac remains open for inspection/reset" in source
    assert "simulation_app.close()" not in source[source.index("while simulation_app.is_running()") : source.index("control.close()")]


def test_mobile_base_override_is_explicit_and_source_usd_is_not_edited():
    config = yaml.safe_load((ROOT / "configs/390f_v2_interactive.yaml").read_text(encoding="utf-8"))
    scene = config["scene"]
    assert scene["mobile_base_enabled"] is True
    assert scene["world_anchor_joint"].endswith("/Joints/lower_world_fixed")
    runtime = (ROOT / "isaac_loader/run_390f_v2.py").read_text(encoding="utf-8")
    assert "anchor.GetPrim().SetActive(False)" in runtime
    assert ".Save(" not in runtime

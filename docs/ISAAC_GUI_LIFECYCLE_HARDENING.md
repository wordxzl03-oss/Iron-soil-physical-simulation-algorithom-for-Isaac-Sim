# Isaac GUI lifecycle hardening

## Root cause and fix

The reproducible flash-close was a `UI_EXCEPTION`, not CUDA, PhysX, terrain, or material physics. Isaac Sim 4.5 `omni.ui.Label` requires a positional text string and rejects `model=SimpleStringModel`. The HUD construction raised `TypeError` before frame zero; the outer finalizer then correctly closed SimulationApp, which looked like a spontaneous shutdown.

Dynamic text is now shown by read-only multiline `StringField` widgets bound once to `SimpleStringModel`. Runtime code updates only model values. There are no direct per-frame `Label.text` assignments and no per-frame window/UI-tree reconstruction.

The application loop is solely `while simulation_app.is_running()`. READY and PAUSED continue updating Kit. Only explicit automated completion, a user window close, an external signal, or a classified exception can leave the loop. Signal callbacks now latch an exit request for the owning loop instead of raising inside Kit's async engine. `application_exit.json` records the exit owner and last state.

## Real GUI evidence

- Production lifecycle soak: READY held 30.005 s with the process alive; actual dig and terrain publication ran; GUI stayed alive after a production `REVERSE_TRAVEL_TIMEOUT`; RESET ALL passed; RESET→READY held 30.001 s; harness then exited explicitly.
- RESET restored payload and Mobile to zero, terrain volume to 1.61e-11 m³, joint positions to 2.19e-8 rad, and base pose exactly.
- Normal `./run_390f_presentation_demo.sh`: `MANUAL_GUI`, auto-start false, auto-exit false, READY stayed open more than 30 s.
- Presentation regression: `./run_390f_presentation_demo.sh --autostart --exit-after-hold` reached its configured 60 s `HOLD` with GPU_RUNTIME/DEVICE, 701×701 at 0.05 m, authoritative H_free dirty-tile visual publication, peak payload 0.31565 m³, mass error 7.73e-12 m³, and zero root-pose writes. It exited with `AUTOMATED_TEST_COMPLETED` and no exception.

## Claim boundary

GUI lifecycle stability and the delivered presentation regression pass. The ordinary production closed-loop soak is PARTIAL, not PASS: its unchanged task state machine reached `REVERSE_TRAVEL_TIMEOUT` at 21.8167 simulation seconds, before dump/post-dump hold. That is an existing production motion/gate issue, not a GUI exit, and this task did not relax its state-machine conditions.

Machine-readable evidence: `outputs/isaac_gui_soak/summary.json`.

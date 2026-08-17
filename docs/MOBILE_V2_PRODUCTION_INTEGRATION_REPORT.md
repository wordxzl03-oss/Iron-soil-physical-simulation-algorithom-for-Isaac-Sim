# MOBILE V2 PRODUCTION INTEGRATION REPORT

Date: 2026-08-13

This is the production-integration report, not a repeat of the standalone
reference report. Gates A and B passed. Gate C stopped at the first hard
physical task failure, so no production-candidate promotion, full cycle, or
GUI run was attempted.

## Required status

```text
STATE_MIGRATION: PASS
AUTHORITATIVE_B_EFF: PASS
CHECKPOINT_B_EFF_PERSISTENCE: PASS
ENTRAINMENT_CONTRACT: PASS
DEPOSITION_CONTRACT: PASS
AIRBORNE_TO_V2: PASS
OWNERSHIP_CONTRACT: PASS
FAILUREZONE_TO_V2: PASS
LARGE_AVALANCHE_TO_V2: PASS
MINISLOPE_SEMANTICS: PASS
GATE_A: PASS
FROZEN_5S: PASS
FROZEN_5S_MAX_MASS_ERROR_M3: 8.881784197001252e-16
FROZEN_5S_ENERGY_STATUS: PASS
PRE_DUMP: FAIL
VISIBLE_NEEDLE_FOREST: NOT_EVALUATED_GATE_C_DID_NOT_REACH_PRE_DUMP
VISIBLE_TRIANGULAR_FINS: NOT_EVALUATED_GATE_C_DID_NOT_REACH_PRE_DUMP
GRID_SCALE_SPIKE_PROLIFERATION: NOT_EVALUATED_GATE_C_DID_NOT_REACH_PRE_DUMP
PRE_DUMP_JH_MAX_M: NOT_CAPTURED
PRE_DUMP_JH_P99_M: NOT_CAPTURED
PRE_DUMP_MOBILE_M3: NOT_CAPTURED
PRE_DUMP_PAYLOAD_M3: NOT_CAPTURED
MASS_CONSERVATION: PASS
ENERGY_CONSISTENCY: PASS_GATE_B; NOT_EVALUATED_GATE_C
PERFORMANCE_RTF: 1.0465270730648888_GATE_B; 0.31818927213590165_GATE_C_PARTIAL_MEAN
PRODUCTION_CANDIDATE: NO
FULL_CYCLE: NOT_RUN
ARREST_FINAL: NOT_RUN
GUI: NOT_RUN
ROOT_REMAINING_LIMITATION: CUT_AND_FILL reached only 0.014497247407260367 m3 payload before its unchanged 12.0 s physical timeout; the required 0.02 m3 phase gain was not met, so breakout/reverse/pre-dump morphology was never reached.
```

The three qualitative morphology fields intentionally remain
`NOT_EVALUATED`, rather than inventing YES/NO values for a state that was not
reached.

## State and exchange closure

Production DEVICE authority is now `z_base`, `b_eff`, `mobile`,
`momentum_x`, and `momentum_y`. `resting` is only a same-allocation alias of
`b_eff`; it is not an independently writable terrain geometry. The host
reference `TerrainState.H_resting_m` is explicitly a source-compatible name
for `b_eff`, and its ledger depth is derived as `b_eff-z_base`.

The compact DEVICE entrainment primitive donor-limits by `b_eff-z_base`,
updates `b_eff -= depth` and `mobile += depth`, and leaves momentum unchanged.
The deposition primitive donor-limits by Mobile thickness, performs the
inverse interface motion, and reports removed Mobile momentum. Gate A measured
zero instantaneous `H_free` error for both exchanges, zero activation momentum
error, zero ledger error, and zero direct `b_eff` restore error.

FailureZone now calls the authoritative entrainment API. LargeAvalanche uses
the same donor-limited interface-motion contract and no longer assigns a
gravity launch velocity during activation; finite-time V2 flux/source physics
owns subsequent acceleration. Airborne still writes only Mobile thickness and
momentum. The accepted conservative-export latch state and export baselines
remain resident and are checkpointed.

MiniSlope's DEVICE field is now named `b_eff` and its declared semantics are
`QUASI_STATIC_BED_TRANSPORT_DYNAMIC_MOBILE_MUST_BE_QUIET`. The production Core
already gates it behind quiet Mobile/Airborne state; it is not required to run
when there is no physical residual seed.

## Gate B evidence

The real 701x701 frozen field was loaded through
`UNCALIBRATED_SNAPSHOT_COMPATIBILITY_ADAPTER`, then advanced continuously at
0.05 m resolution to 5 s by the production `GpuBulkOperatorChain` using
`GPU_WARP_MOBILE_V2_PRODUCTION_SHARED_STATE`.

```text
uniform-control mass error:             0.0 m3
exact terrain-weight residual:          -8.881784197001252e-16 m3
minimum Mobile thickness:               0.0 m
H_free max 4-neighbor jump:             1.2934292275930663 m
H_free p99 4-neighbor jump:             0.05374707625461808 m
Mobile volume:                          3.0804917641483334 m3
moving Mobile volume:                   0.0 m3
maximum speed:                          0.0012605766434034503 m/s
E_density_normalized total:             11.069315288052074
E_physical total:                       15164.96194463134 J
wall time:                              4.777707264998753 s
RTF:                                    1.0465270730648888
```

Against the accepted standalone result, max jump improved by
0.0007195196124993863 m and p99 improved by 0.000045969861516788335 m.
Extrema count was 1400 versus 1393; above/below-envelope volumes were
0.13090841735079745/0.23609386770756804 m3 versus
0.130495408443383/0.235751111989263 m3. These are within the controlled 2%
non-regression band and do not constitute a new dominant mode.

The first numerical difference is explained: the standalone frozen evidence
used its default basal coefficient 0.55, while the immutable production iron
ore scenario specifies 0.35. The production run retained 0.35 as required; no
parameter was tuned to reproduce the standalone trace.

## Gate C hard stop

Exactly one fresh production 390F run was launched with GPU_RUNTIME / DEVICE,
701x701, and 0.05 m spacing. It reached APPROACH, PENETRATE, and CUT_AND_FILL,
then stopped on the existing physical timeout:

```text
failure: CUT_AND_FILL_TIMEOUT
simulation time: 17.700000923126936 s
payload: 0.014497247407260367 m3
Mobile: 4.772207460822411 m3
activated: 4.619549241054479 m3
peak soil force: 288473.46021197806 N
max absolute conservation error: 1.546140993013978e-11 m3
```

The state-machine threshold, trajectory, material, dt, and physics were not
changed. Since ALIGN_DUMP was never reached, the release-before-dump capture
was correctly not emitted. Per the sprint gate rules, full-cycle and GUI tests
were not run.

## Evidence

- `outputs/mobile_v2_production/mobile_v2_production_gate_ab.json`
- `outputs/390f_v2/production_gpu_core_acceptance.json`
- `outputs/390f_v2/interactive_runs/run_1786613917/runtime_failure.json`
- `outputs/390f_v2/interactive_runs/run_1786613917/gpu_real_one_cycle_acceptance.json`

## Classification boundary

```text
Numerical architecture: LITERATURE_INFORMED_REDUCED_ORDER
b_eff lifecycle: CONSERVATION_BASED_ENGINEERING_CLOSURE
K=0.45: ENGINEERING_CLOSURE_UNCALIBRATED
iron-ore physical calibration: NOT_YET_PHYSICALLY_CALIBRATED
```

The integration is not promoted while Gate C is incomplete.

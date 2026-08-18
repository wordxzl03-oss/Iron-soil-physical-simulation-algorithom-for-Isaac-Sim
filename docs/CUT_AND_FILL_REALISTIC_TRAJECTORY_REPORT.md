# CUT_AND_FILL realistic trajectory report

## Result

The frozen production trajectory is **PUSH_PENETRATE_DRAG**: its CUT target held
the bucket at -60° and moved stick from about 16° toward 6.33°, so principal
failure preceded positive curl/retraction.  The production-default replacement
is a continuous A–E curl-scoop consumed by the unchanged bounded actuator.

The valid normal-boundary replay reached PRE_DUMP.  Net payload increased
2.659× and gross capture ratio increased
12.525× while maximum penetration fell.
This supports H1 without changing soil physics.

## Old versus new

| Metric | old | new |
|---|---:|---:|
| MAX_PENETRATION_M | 1.903033167 | 0.6560070643 |
| ACTIVE_CUT_DURATION_S | 12.00000063 | 4.933333591 |
| PENETRATION_DOMINATED_FRACTION | 0.3027777778 | 0.5202702703 |
| SIMULTANEOUS_STICK_RETRACT_BUCKET_CURL_FRACTION | 0 | 0.6418918919 |
| PEAK_BUCKET_CURL_RATE_RAD_S | 0.001294453163 | 0.3281877339 |
| MEAN_BUCKET_CURL_RATE_DURING_ACTIVE_CUT_RAD_S | -0.0005510092625 | 0.1419161871 |
| MEAN_STICK_RETRACTION_RATE_DURING_ACTIVE_CUT_RAD_S | -0.02779790735 | 0.01432707698 |
| FAILUREZONE_R2M_M3 | 4.614720164 | 1.001389061 |
| TOTAL_R2M_M3 | 4.781875632 | 1.015189061 |
| MOBILE_MOUTH_P95_M3 | 0.005284910638 | 0.0003963251815 |
| MOBILE_FRONT_P95_M3 | 2.98689548 | 0.7850022196 |
| MEAN_POSITIVE_RELATIVE_NORMAL_SPEED_M_S | 0.01878238618 | 0.353564121 |
| P95_POSITIVE_RELATIVE_NORMAL_SPEED_M_S | 0.05254870143 | 0.6200688022 |
| CANDIDATE_MOUTH_FLUX_M3 | 0.01430071544 | 0.03802625208 |
| ACCEPTED_MOUTH_FLUX_M3 | 0.01430071544 | 0.03802625208 |
| NET_PAYLOAD_GAIN_M3 | 0.01430071544 | 0.03802625208 |
| GROSS_CAPTURE_RATIO | 0.002990607984 | 0.03745731071 |
| PEAK_SOIL_FORCE_N | 288473.4602 | 216159.1912 |
| EFFORT_SATURATED_STEP_FRACTION | 0.02222222222 | 0.02142857143 |
| MAX_MASS_ERROR_M3 | 6.821210263e-12 | 2.137312549e-11 |
| PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N | 0 | 0 |

The common comparison interval uses `penetration>0 AND Mobile-within-2m>0`
because the historical v1 archive retained total FailureZone R2M but not its
per-step values.  The candidate additionally reports the stricter per-step
FailureZone-active interval in the JSON.

## Temporal ordering

Candidate peak FailureZone R2M occurred at simulation t=
7.500000s,
peak mouth overlap at 7.966667s,
peak mouth flux at 8.433334s,
peak curl rate at 14.000001s,
and peak penetration at 7.500000s.

## PRE_DUMP morphology and conservation boundary

- PRE_DUMP reached: **YES**
- Visible needle forest: **YES**
- Visible triangular fins: **YES**
- Grid-scale spike proliferation: **YES**
- max / p99 H_free neighbor jump: 0.756472999 / 0.03553 m
- CUT/CURL max mass error: 2.13731255e-11 m³
- full replay max mass error: 0.000219067992 m³

The trajectory/intake path remains conservative to numerical precision.  The
full replay is not a conservation pass: error begins only when TrackSoil first
activates during REVERSE.  This frozen-module regression is recorded, not
silently attributed to the curl-scoop or repaired in this task.

Dynamic Mobile momentum reaction remains exactly zero under real active
contact, so the required next P0 label is
`TOOL_MOBILE_MOMENTUM_COUPLING_REQUIRES_CAUSAL_AUDIT`.

## Replay and regression evidence

- valid production replay: `run_1786937073`, 390F / GPU_RUNTIME / DEVICE /
  701×701 / dx=0.05 m, stopped at the existing pre-release PRE_DUMP boundary;
- focused trajectory/runtime regression: 18 passed;
- full CPU suite: 279 passed, 16 skipped, 57 subtests passed;
- three pre-existing unrelated failures remain: two stale Phase-A manifest
  checks for `runtime/__init__.py`, and the existing Phase-H width monotonicity
  assertion.  None of those files/models were changed here.

The historical CUT checkpoint was restored once for diagnosis, but it lacks
root z/roll/pitch and actuator internal target velocity.  Its initial
penetration became 1.53 m instead of the frozen 0.23 m, so that interrupted
state is explicitly excluded.  The accepted result reran the unchanged normal
production prelude to obtain a physically complete PENETRATE→CUT boundary;
no root/link pose write or teleport was added.

## Artifacts

- `outputs/mobile_v2_production/cut_fill_realistic_trajectory_report.json`
- `outputs/mobile_v2_production/cut_fill_realistic_trajectory_timeseries.json`
- `outputs/mobile_v2_production/cut_fill_realistic_trajectory_pre_dump.npz`
- `outputs/mobile_v2_production/cut_fill_realistic_trajectory_alignment.png`
- `outputs/mobile_v2_production/cut_fill_realistic_trajectory_pre_dump_morphology.png`

# Mobile V1 accelerated closure report

`GATE_A_5S: FAIL`

Both DEVICE branches restored the same frozen `t=3.75 s` fields and ran in one
invocation through the `0.5 / 2 / 5 s` horizons. The production default
remained `CURRENT_CENTERED`; only the validation branch selected
`EXPERIMENTAL_FACE_CONSISTENT`.

The first causal failure is candidate mechanical-energy creation at the first
recorded horizon. With effectively zero tool work (`1.99e-12 J` over the full
window), candidate mechanical energy rose by `2519.96 J` at `0.5 s`; CURRENT
fell by `895.18 J`. The diagnostic allowance was `0.371 J`. Since identical
restored state and operator chain were used and the source stencil was the only
branch difference, this is attributed to the experimental face-source closure.

The later metrics reinforce rather than rescue that failure:

| 5 s metric | CURRENT | Candidate |
|---|---:|---:|
| max four-neighbor `H_free` jump (m) | 1.62594 | 2.05373 |
| p99 four-neighbor `H_free` jump (m) | 0.073384 | 0.089123 |
| R2M from prior-M2R regions (m3) | 1.51323 | 2.04126 |
| new-extrema count | 3066 | 6476 |
| above-envelope volume (m3) | 0.38601 | 1.67107 |
| below-envelope volume (m3) | 0.33812 | 1.16060 |

Mass and positivity still pass: worst candidate ledger error is
`3.00e-11 m3` and minimum `h_mobile` is zero. They do not compensate for the
energy and morphology failures.

Gate B, the full cycle, and GUI were not run because the Gate A contract
requires stopping at the first causal failure. The experimental source is not
a production candidate, and the production default remains frozen.


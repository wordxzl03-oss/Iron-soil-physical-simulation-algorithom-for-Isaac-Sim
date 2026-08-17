# V3 runtime performance audit

## Result

After physics closure, the same frozen 701×701 GPU checkpoint advances 5.0 simulation seconds in 1.9287 wall seconds (`RTF=2.5924`). The equivalent pre-idle path took 13.4637 seconds (`RTF=0.3726`), for a 6.98× wall-clock speedup. First `ARREST_FINAL` remains 1.35 simulation seconds and maximum mass error remains 2.09e-11 m³.

## Causal profile

Before closure, the dominant cost was algorithmic: a false R2M/M2R event every `dt`. After closure but before optimization, the core still launched Mobile, LargeAvalanche connected-region reductions, two full mass-ledger reductions, deposition, and residual scheduling on every quiet step.

The event-driven idle branch now applies only after authoritative settled diagnostics show no airborne parcels, no residual frontier, and Mobile below the existing settled threshold. It advances device/metadata time and reuses the immutable scalar reservoir/avalanche snapshot. Any dig, dump, TrackSoil R2M, airborne activity, Mobile activity, or residual tile invalidates the branch. Launcher/HUD reservoir reads reuse the core's current scalar snapshot rather than launch a duplicate full-field reduction.

Of 300 profile steps, 219 used this idle path. Active-step totals show Mobile transport, LargeAvalanche, and the two ledger reductions as the remaining major core costs. Residual projection is negligible for this case. No full-field H2D or D2H occurred in the profiled normal path; cumulative compact transfers were 7,560 H2D bytes and 201,364 D2H bytes.

The profile is core-only and post-dump. GUI render, PhysX, dirty visual mesh publication, contact chunks, HUD, approach, digging, FailureSurface, and SoilForce are reported by the separate Isaac run and are not folded into the core-only speedup claim.

Machine-readable timing distributions (mean/p95/max/total and percent of core wall): `outputs/runtime_profile/summary.json`.

# Capability matrix

| Capability | Status | Reality |
|---|---|---|
| Bucket excavation | SUPPORTED | One registered bucket, V3 production path |
| Tool reaction force | SUPPORTED | World force and application point |
| Tool torque | SUPPORTED | Total origin torque plus residual couple |
| Payload volume | SUPPORTED | Authoritative payload reservoir |
| Payload mass | SUPPORTED | Uniform scenario density × volume |
| Terrain deformation | SUPPORTED | Shared `H_free` at configured resolution |
| Mobile soil | SUPPORTED | CPU and Warp production backends |
| Deposition | SUPPORTED | Incremental production operator |
| Mass ledger | SUPPORTED | Conservative volume ledger |
| GPU runtime | SUPPORTED | Requires Isaac Warp/CUDA environment |
| DEVICE authority | SUPPORTED | No normal full-field host terrain |
| Isaac visual terrain sync | SUPPORTED | Dirty `H_free` chunks |
| Isaac collision/contact sync | PARTIAL | Dirty static-mesh recook, caller scheduled |
| Left track contact | PARTIAL | Footprint-driven terrain transfer; no wrench |
| Right track contact | PARTIAL | Footprint-driven terrain transfer; no wrench |
| Track normal support | NOT_SUPPORTED | No normal reaction closure |
| Track traction | NOT_SUPPORTED | Vehicle traction remains external |
| Track sinkage | SUPPORTED | Engineering reduced-order, uncalibrated |
| Track rutting | SUPPORTED | Conservative rut-to-shoulder transfer |
| Independent belt speed | PARTIAL | Mapped to existing effective surface velocity |
| Longitudinal slip | PARTIAL | Effective velocity difference only; no explicit state input |
| Lateral slip | NOT_SUPPORTED | No lateral constitutive closure |
| Compaction/density evolution | NOT_SUPPORTED | Uniform density only |
| Wheel physics | NOT_SUPPORTED | Out of alpha scope |
| Multi-tool support | NOT_SUPPORTED | One bucket production core |
| Flow/arrest final closure | NOT_VALIDATED | `FLOW_ARREST_CLOSURE_NOT_YET_DEMONSTRATED` |
| RL observation API | SUPPORTED | Read-only physical feedback and aggregation |


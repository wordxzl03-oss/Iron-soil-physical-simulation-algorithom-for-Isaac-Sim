# Failure Zone theory trace

## Scope and assumptions

This module is a quasi-static, reduced-order failure-geometry model. It is not
a DEM model, a complete Fundamental Earthmoving Equation (FEE) implementation,
or a compaction model. The state ledger conserves volume and converts volume to
estimated mass with one assumed constant bulk density. Compaction, dilation,
moisture-dependent density and pore-pressure evolution are not represented.

Primary sources:

- O. Luengo, S. Singh and H. Cannon, *Modeling and Identification of
  Soil-tool Interaction in Automated Excavation*, IROS 1998, DOI
  10.1109/IROS.1998.724873. [Author-hosted paper](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/luengo_o_1998_1/luengo_o_1998_1.pdf)
- M. Servin, T. Berglund and S. Nystedt, *A multiscale model of terrain
  dynamics for real-time earthmoving simulation*, Advanced Modeling and
  Simulation in Engineering Sciences 8, 1 (2021), DOI
  10.1186/s40323-021-00196-3. [Primary manuscript](https://arxiv.org/pdf/2011.00459)
- G. McKyes and O. S. Ali, *The cutting of soil by narrow blades* (1977),
  DOI 10.1016/0022-4898(77)90001-5.
- Z. X. Zhang and R. L. Kushwaha, *A nonlinear mechanical model of soil-tool
  interaction* (1995), DOI 10.1016/0167-1987(95)00466-6.

## Trace table

| Component | Classification | Source and equation/section | Code | Deviation and validation |
|---|---|---|---|---|
| Sloped-ground force factors `Nw`, `Nc`, `Nq` | `DIRECT_LITERATURE_MODEL` | Luengo et al. (1998), §3.1, equations (5)–(6) | `FailureZoneModel._solve_strip()` | Implemented with the paper's angle variables `alpha`, `rho`, `delta`, `phi` and SI units. Formula-domain singularities are rejected instead of hidden by clamps. Unit/sensitivity tests exercise finite admissible domains. |
| Critical failure angle `beta` | `LITERATURE_BASED_REDUCED_ORDER` | Luengo §3.1–§5 treats a planar failure surface and identifies `beta`; McKyes & Ali evaluate trial wedges and minimize predicted draft; Zhang & Kushwaha minimize total soil resistance | `_solve_strip()` coarse bracket plus golden-section minimization | Luengo identifies `beta` from force data and does not prescribe this online minimizer. The implementation closes that unavailable parameter by minimizing its predicted quasi-static resistance. Boundary-hit rate remains observable. |
| Planar active wedge | `LITERATURE_BASED_REDUCED_ORDER` | Luengo §3.1 static planar wedge; Servin et al. §2.4, equations (20)–(22), active zone bounded by cutting edge, failure surface and separation plane | `FailureStripGeometry.cross_section_vertices_terrain_m` | Each lateral strip is a triangular prism. Servin's equation (22) is not substituted because this implementation solves `beta` from the Luengo force equilibrium. Polygon and positive-extent invariants are tested. |
| Terrain intersection of failure plane | `LITERATURE_BASED_REDUCED_ORDER` | Same planar-wedge construction | `L = d / (tan(beta) - tan(alpha))`, area `d L / 2`, volume `area * width` | Assumes locally planar terrain within a strip. Curvature and transverse shear are neglected. Analytical triangle/area/volume identities are tested. |
| Strip decomposition | `LITERATURE_BASED_REDUCED_ORDER` | Servin et al. §2.4 describes parallel active-zone wedges | `FailureZoneModel.compute()` | Independent 2-D strips neglect cross-strip shear; their lateral bounds partition the intersecting tool span. |
| Separation-plane/rake direction from bucket bottom | `LITERATURE_BASED_REDUCED_ORDER` | Luengo equation (6) requires rake `rho`; Servin Complex Digging Tools defines the bottom as primary separation plate | `BucketGeometryDescriptor.bottom_plate_polygon_local` → `ToolState` → `ToolTerrainIntersection.separation_plane_direction_terrain` | L1 uses the explicit/marker-derived flat bottom tangent. Tool-frame `+Z` survives only as the named `TOOL_PLUS_Z_LEGACY_FALLBACK` when no bucket geometry exists; high-quality geometry acceptance requires its flag to be false. Rake sensitivity remains tested and the chosen angle is stored per strip. |
| Conservative Heightmap mapping | `ENGINEERING_APPROXIMATION` | Servin §2.4 makes resting mass conform to the failure surface before conservative active-soil evolution | `_rasterize_strip_wedge()` | Every vertex control polygon is clipped against the same analytical strip footprint. Because wedge thickness is linear, overlap volume is integrated exactly at the overlap centroid and divided by the existing topology weight. No seed projection or `max`-merged second wedge exists. Resolution and volume tests cover this map. |
| Terrain/boundary clipping | `ENGINEERING_APPROXIMATION` | Numerical domain requirement | `boundary_clipped`, `availability_clipped`, per-vertex requested/activated volumes | Full, requested-in-domain and actually available volumes remain separately observable. Soil-force static terms are scaled by the same strip's activation ratio. |
| Density, friction, cohesion and surcharge values | `UNCALIBRATED_PARAMETER` | Variables appear in Luengo equations (1), (5)–(6) | `MaterialScenario`, `FailureZoneConfig.surcharge_pa` | Scenario values are not claimed as measured iron-ore-powder truth. Surcharge uses the pressure form `q d w Nq`; Luengo's swept-volume surcharge `Vs gamma g (Nq-1)` is unavailable from the current state and is not claimed to be implemented. |
| Soil-force feedback | `LITERATURE_BASED_REDUCED_ORDER` plus `CONSERVATION_BASED_ENGINEERING_MODEL` | Luengo static FEE factors; Servin active-soil mass/momentum coupling | `SoilForceModel.compute()`, `MobileMomentumBudget` | It consumes the exact `FailureStripGeometry` terms. The former arbitrary speed-squared coefficient is removed; the dynamic reaction is the negative of measured tool-to-mobile impulse over the explicit timestep. Direction/resultant deviations are traced in `SOIL_FORCE_THEORY_TRACE.md`. |

## Single-geometry data flow

```text
ToolTerrainIntersection + local terrain + uncalibrated material
                         |
                         v
             critical FailureStripGeometry
               /                         \
              v                           v
polygon/control-volume activation    SoilForceModel input
              |
              v
existing Resting -> Mobile conservative transfer and ledger
```

`candidate_intersection_volume_m3` is only the swept-contact trigger. It is not
the wedge volume and is therefore expected to differ from both the analytical
wedge and the explicitly clipped activation volume.

## Applicability limits

The planar, quasi-static model is intended for low-speed real-time interaction.
It omits curved/nonlocal failure surfaces, transverse shear, strain-rate effects,
large inertial granular flow, compaction and cohesive fracture. Servin et al.
also identify the planar active-zone approximation as a limitation, especially
for strongly cohesive soil. Dynamic redistribution begins only after activation
in the existing Mobile Layer solver.

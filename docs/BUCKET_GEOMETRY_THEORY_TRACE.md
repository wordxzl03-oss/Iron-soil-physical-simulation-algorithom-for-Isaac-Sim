# Bucket geometry and conservative intake theory trace

## Scope and claim boundary

This phase implements **geometry-consistent, conservative reduced-order bucket
intake**. It does not claim fully validated granular bucket filling physics. The
Mobile Layer has only column thickness `h_m(x,y)` and horizontal momentum, so
the intake model deliberately remains 2.5-D; it does not invent a vertical
density or velocity field.

Primary sources:

- M. Servin, T. Berglund and S. Nystedt, *A multiscale model of terrain
  dynamics for real-time earthmoving simulation* (2021), especially Complex
  Digging Tools/Fig. 4. [DOI](https://doi.org/10.1186/s40323-021-00196-3),
  [primary manuscript](https://arxiv.org/pdf/2011.00459).
- C. J. Coetzee and D. N. J. Els, *The numerical modelling of excavator bucket
  filling using DEM* (2009). [DOI](https://doi.org/10.1016/j.jterra.2009.05.003).
- A. Svanberg et al., *Full-scale simulation and validation of bucket filling
  for a mining rope shovel by using a combined rigid FE–DEM granular material
  model* (2021). [DOI](https://doi.org/10.1007/s40571-020-00372-z).
- K. Aoshima and M. Servin, *Examining the simulation-to-reality gap of a wheel
  loader digging in deformable terrain* (2025).
  [DOI](https://doi.org/10.1007/s11044-024-10005-5),
  [primary manuscript](https://arxiv.org/pdf/2310.05765).

## Trace table

| Component | Paper | Section/model | Classification | Implementation | Deviation |
|---|---|---|---|---|---|
| Cutting edge, parallel top edge and penetration direction | Servin et al. 2021 | Complex Digging Tools, Fig. 4 | `DIRECT_LITERATURE_MODEL` for the semantic definitions | `BucketGeometryDescriptor.cutting_edge_local`, `top_edge_local`, `penetration_direction_local` | Dimensions come from markers or an explicit profile, not from the paper. Teeth and a curved multi-section edge are not resolved. |
| Bottom plate as primary separation geometry | Servin et al. 2021 | Complex Digging Tools, primary separation plate | `LITERATURE_BASED_REDUCED_ORDER` | Validated planar `bottom_plate_polygon_local`; its tangent supplies `ToolTerrainIntersection.separation_plane_direction_terrain` | A single flat plate replaces a curved or piecewise real bottom. Legacy `+Z` remains only as an explicit no-geometry fallback and is rejected by high-quality acceptance. |
| Mouth plane, polygon, inward normal and local frame | Servin et al. 2021 | Face connecting cutting and top edges | `LITERATURE_BASED_REDUCED_ORDER` | Planar polygon bounded by lip/cutting and top edge; SVD plane fit, deterministic inward normal, area/centroid/frame invariants | Servin describes separation faces rather than this exact software polygon contract. |
| Closed bucket interior, back and side walls | Servin et al. 2021; Coetzee & Els 2009 | Inner shape; physical bucket profile | `LITERATURE_BASED_REDUCED_ORDER` | One simple y–z interior profile laterally extruded across cutting-edge width into consistently wound watertight triangles | Real curvature, teeth, nonuniform cross sections and wall thickness are omitted. |
| Geometric capacity | Coetzee & Els 2009 | Physical bucket geometry and maximum holding capacity | `LITERATURE_BASED_REDUCED_ORDER` | Exact profile shoelace area × width, cross-checked against positive closed-mesh volume | This is closed/struck geometric volume. Rated/heaped capacity remains a separate value; effective solver capacity is conservatively bounded by both when rated capacity exists. |
| Marker geometry source | Svanberg et al. 2021 | Measured/laser-scanned real bucket geometry as physical model input | `ENGINEERING_APPROXIMATION` | Validated eight-marker contract → Tool Frame → reduced-order closed profile; `geometry_source=MARKERS` | It is not a laser scan or mesh reconstruction. It captures only the contract landmarks. |
| USD/mesh plus semantic-marker source | Svanberg et al. 2021 | Real 3-D bucket geometry as physical input | `ENGINEERING_APPROXIMATION` | Audited 390F mesh vertices identify cutting/top/rear landmarks and an ordered concave side profile; `geometry_source=USD_MESH_MARKERS`; the profile is extruded across measured interior width | The marker indices are locked to the asset SHA-256. It is CAD-constrained but still replaces a varying 3-D interior by one cross-section; see `REAL_390F_BUCKET_INTEGRATION.md`. |
| 2.5-D mouth control-surface flux | Compatible with Servin 2021 and Coetzee & Els 2009 filling mechanism; not their equation | Material enters through the bucket opening | `CONSERVATION_BASED_ENGINEERING_MODEL` | `∫_Γ h_overlap [(u_m-v_mouth)·n_xy]_+ dℓ`, exact projected lip/control-rectangle clipping, angular mouth velocity and timestep | Not a 3-D area integral and not a DEM flow law. Vertical velocity, granular shear and pore-scale flow are absent. No capture/efficiency coefficient exists. |
| Donor and capacity limiting | Volume conservation and common interior contract | Numerical finite-volume constraint | `CONSERVATION_BASED_ENGINEERING_MODEL` | Each control volume cannot donate more than `h_m w_i`; global admitted volume cannot exceed remaining effective interior capacity | Simultaneous subcell replenishment is not reconstructed during one timestep. CFL-scale time refinement is therefore required. |
| Intake centroid and payload COM update | Conservation of first moment | Reduced-order state transfer | `CONSERVATION_BASED_ENGINEERING_MODEL` | Admitted-volume-weighted terrain centroid is transformed to bucket frame and mixed with existing payload COM | No internal free-surface/dead-load distribution is evolved. |
| Initial flow, active flow and dead load phenomenology | Coetzee & Els 2009 | Bucket-filling observations | `NOT_IMPLEMENTED_THIS_PHASE` | Incoming Mobile Layer transfers only to aggregate `PayloadState` | No laminar-layer ratio, dead-load ratio or shear-zone rule is invented. |
| Fill-dependent secondary separation plate | Servin et al. 2021 | Full-bucket secondary separation geometry | `NOT_IMPLEMENTED_THIS_PHASE` | Geometry contract preserves mouth, interior and fill capacity for a later fill-state model | Primary bottom separation geometry is used for the empty/current aggregate bucket; separation normal is not interpolated with fill ratio. |
| Future validation observables | Svanberg et al. 2021; Aoshima & Servin 2025 | Loaded mass/fill factor, trajectory, forces and work | `LITERATURE_BASED_REDUCED_ORDER` interface requirement | Geometry source/quality, admitted volume, centroid and payload mass-compatible state remain observable | No iron-ore calibration, full Isaac runtime comparison, force/work or sim-to-real validation is claimed here. |
| Parameterized explicit profile dimensions | Repository YAML scenario values | Width/depth/rear/interior heights | `UNCALIBRATED_PARAMETER` | `bucket_small/medium/large.yaml` build `EXPLICIT_PROFILE`, `REDUCED_ORDER` geometry | Values are not asserted to be manufacturer measurements or calibrated iron-ore truth. |
| Legacy inferred profile | Historical `ToolDescriptor` arrays | Compatibility only | `ENGINEERING_APPROXIMATION` | Explicit `geometry_source=LEGACY_FALLBACK`, `geometry_quality=APPROXIMATE`; production intake rejects it by default | Exists only to keep old consumers diagnosable. It cannot silently pass geometry acceptance. |

## One authoritative data flow

```text
USD mesh (reserved) / validated markers / explicit profile / flagged fallback
                                  |
                                  v
                    BucketGeometryDescriptor
       cutting -- bottom -- lip -- mouth -- top -- interior -- capacity
             |                    |                    |
             v                    v                    v
 ToolTerrainIntersection   conservative intake   Payload capacity
             |
             v
 existing Failure Zone beta solver and conservative rasterizer (unchanged)
```

`mouth_depth_m` now helps define the explicit interior profile only. It is not
an intake capture depth. `sampling_radius_m`, `mouth_height_m`, capture ratio and
intake efficiency do not participate in admitted volume.

## Numerical evidence

Run:

```bash
.venv/bin/python -m pytest -q tests/test_bucket_geometry_intake.py
.venv/bin/python tools/run_bucket_geometry_acceptance.py
```

The machine-readable result is `outputs/bucket_geometry_acceptance.json`. It
includes analytic perpendicular/oblique/partial-overlap errors, mobile–payload
conservation, capacity overflow, projected overlap, intake first-moment spatial
convergence, prescribed time-varying flux convergence, and formal negative
flags for the old sampling band and tool-`+Z` fallback.

Use `configs/project_25m_l1.yaml` for the 701×701/0.05 m modular 25 m scenario.
The hashed `configs/project_25m.yaml` remains the frozen L0 replay input and is
not eligible for high-quality intake acceptance.

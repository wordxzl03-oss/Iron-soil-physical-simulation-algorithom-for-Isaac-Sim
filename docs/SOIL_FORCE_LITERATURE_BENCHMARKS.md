# SoilForce literature benchmarks

## Executed external benchmark

The quantitative benchmark is Section 3.1 of Obermayr, Vrettos, Kleinert &
Eberhard, *A Discrete Element Method for assessing reaction forces in
excavation tools*, Fraunhofer ITWM Report 228 (2013). The related peer-reviewed
article is *Prediction of draft forces in cohesionless soil with the Discrete
Element Method*, DOI `10.1016/j.jterra.2011.08.003`.

Reproducibility is classified `FULLY_REPRODUCIBLE` for the current quasi-static
horizontal-draft comparison because the report publishes:

- a straight vertical blade, widths 0.20 and 0.30 m;
- four test depths from 0.10 through 0.25 m used in Figures 9--10;
- washed uniformly graded 1.0--2.2 mm sand;
- bulk density 1,440 kg/m³, internal friction 35 degrees, cohesion intercept
  1 kPa, and soil--tool friction coefficient 0.29;
- velocity below 0.5 m/s, explicitly described as quasi-static;
- averages over 0.5--1.5 m travel and three test repetitions.

Experimental means were digitized from the three measured crosses at each
condition in Figures 9--10. They are not an undisclosed raw table. The benchmark
stores digitization uncertainty of 35 N for the 0.20 m blade and 60 N for the
0.30 m blade. Consequently the reported discrepancy combines model error,
experimental scatter and figure digitization.

| Width (m) | Depth (m) | Experiment (N) | Model (N) | Absolute relative error |
|---:|---:|---:|---:|---:|
| 0.20 | 0.10 | 280 | 236.46 | 15.5% |
| 0.20 | 0.15 | 505 | 425.14 | 15.8% |
| 0.20 | 0.20 | 820 | 660.74 | 19.4% |
| 0.20 | 0.25 | 1310 | 943.26 | 28.0% |
| 0.30 | 0.10 | 610 | 354.69 | 41.9% |
| 0.30 | 0.15 | 780 | 637.71 | 18.2% |
| 0.30 | 0.20 | 1350 | 991.11 | 26.6% |
| 0.30 | 0.25 | 2200 | 1414.88 | 35.7% |

Aggregate results are MARE 25.14%, RMSE 353.34 N, maximum absolute error
785.12 N and zero force-cap hits. The increasing underprediction at the larger
depths is a material limitation, not calibrated away. No published or site-
calibrated error threshold was selected, so the status is
`EXTERNAL_BENCHMARK_EXECUTED_WITH_QUANTIFIED_MODEL_DISCREPANCY`, not an accuracy
pass or field validation.

The model used exactly the published FEE material inputs, a 90-degree rake,
zero surcharge and no active-momentum impulse. It compared horizontal draft;
the report's vertical reaction is outside this first benchmark.

## Sources reviewed but not used for formal error

| Source | Classification | Missing item for this implementation |
|---|---|---|
| Luengo, Singh & Cannon (1998), DOI `10.1109/IROS.1998.724873` | `PARTIALLY_REPRODUCIBLE` | Equations and identification method are published, but the individual force histories, synchronized bucket trajectories/terrain and final per-dig parameter tuples needed to replay the plotted experiments are not. |
| Coetzee & Els (2009), DOI `10.1016/j.jterra.2009.05.003` | `PARTIALLY_REPRODUCIBLE` | Valuable bucket-filling geometry/phenomenology, but not adopted as a force benchmark here because the current heightmap model cannot reproduce its DEM internal-flow observables and no parameters are guessed. |
| Servin, Berglund & Nystedt (2021), DOI `10.1186/s40323-021-00196-3` | `PARTIALLY_REPRODUCIBLE` | Provides the active-zone/momentum architecture, not a raw force trace for the present 390F case. |

Internal conservation, sensitivity and invariant tests remain numerical tests;
they are never counted as external validation.

## Finite-width discrepancy diagnosis

The current strip implementation evaluates the same plane-failure mechanics
independently across lateral strips.  For the Obermayr straight-blade fixture,
depth, rake, slope and material inputs are laterally uniform.  Every strip
therefore has the same force per unit width and summation necessarily gives
approximately `F ∝ width`.  Raster clipping can perturb this relation near grid
boundaries, but it is not three-dimensional side mechanics.

That structure omits the lateral failure surfaces and side shear that become a
larger fraction of total resistance for a narrow blade.  This is consistent
with the direction of the measured discrepancy: the present model increasingly
underpredicts the wider/deeper Obermayr cases, most severely the 0.30 m blade.
It does not prove one omitted term is solely responsible; cohesion/adhesion,
failure-surface selection, remoulding and a critical-depth transition remain
possible contributors.

The following primary-source records were checked:

- McKyes & Ali (1977), *The cutting of soil by narrow blades*, describes a
  three-dimensional straight failure pattern and force factors dependent on
  width/depth, rake and friction.
- Zhang & Kushwaha (1995) describes a modified three-dimensional McKyes--Ali
  model obtained by minimizing total resistance.
- Bennett et al., *Integration of digging forces in a multi-body-system model
  of an excavator*, describes an FEE plus cavity-expansion construction across
  stages of a digging cycle.

The accessible primary records did not expose the complete equations,
definitions and input set needed for an exact independent implementation.
Consequently the repository deliberately contains no guessed
`FiniteWidth3DResistanceModel`, width gain, or cavity-expansion coefficient.
Their statuses are respectively
`BLOCKED_FULL_PRIMARY_EQUATION_NOT_RECOVERED` and
`BLOCKED_FULL_PRIMARY_EQUATION_AND_INPUT_SET_NOT_RECOVERED`.

The implemented physical decomposition therefore remains:

```text
F_total = F_admissible_plane_failure + F_measured_mobile_momentum
```

with the lateral/side and cavity-expansion terms explicitly absent, rather
than silently absorbed into calibration.

## Reproduction

```bash
.venv/bin/python tools/run_soil_force_literature_acceptance.py
.venv/bin/python -m pytest -q tests
```

Inputs are in `configs/literature/obermayr_2013_blade_benchmark.json`; the
complete cases and metrics are in
`outputs/soil_force_literature_acceptance.json`.

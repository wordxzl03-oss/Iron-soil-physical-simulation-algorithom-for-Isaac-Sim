# Physics model and claim boundary

The package freezes the existing production V3 chain: tool/terrain sweep,
Failure Surface V3, FEE resistance, localized momentum coupling, conservative
Resting→Mobile transfer, Mobile dynamics, bucket intake/payload, retention,
airborne transport, deposition, cohesive large-avalanche transition and compact
residual MiniSlope. TrackSoil conservatively moves rut volume to Mobile
shoulders.

Terms retain their classifications: `PAPER_DIRECT`,
`LITERATURE_INFORMED_REDUCED_ORDER`, `ENGINEERING_CLOSURE`, and
`CONSERVATION_BASED_ENGINEERING_CLOSURE`. The assembled reference scenario is
`NOT_YET_PHYSICALLY_CALIBRATED`.

Bucket, tracks, Mobile and deposition share one state authority. No component
uses the visual USD mesh as physics input. The free surface is
`H_free = H_resting + h_mobile`.


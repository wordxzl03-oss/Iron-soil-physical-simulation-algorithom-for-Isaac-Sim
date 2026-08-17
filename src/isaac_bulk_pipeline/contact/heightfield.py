"""Capability decision for a future native PhysX height-field backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal


CapabilityState = Literal[
    "unavailable_public_api",
    "authoring_only",
    "update_unverified",
    "available",
]


@dataclass(frozen=True)
class HeightFieldCapabilityStatus:
    runtime_label: str
    state: CapabilityState
    public_schema_symbols: tuple[str, ...]
    public_authoring_verified: bool
    public_dynamic_update_verified: bool
    selected_backend: str
    reason: str

    @property
    def usable(self) -> bool:
        return self.state == "available"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_heightfield_capability(
    *,
    runtime_label: str,
    public_schema_symbols: Iterable[str],
    public_authoring_verified: bool = False,
    public_dynamic_update_verified: bool = False,
) -> HeightFieldCapabilityStatus:
    """Choose HeightField only after public authoring *and* update probes pass.

    Symbol discovery alone is intentionally insufficient.  This prevents a
    private PhysX binding or an import-only helper from becoming a production
    dependency.
    """

    symbols = tuple(sorted({str(value) for value in public_schema_symbols}))
    if not symbols:
        state: CapabilityState = "unavailable_public_api"
        reason = "no public USD/PhysxSchema HeightField authoring symbol was found"
    elif not public_authoring_verified:
        state = "authoring_only"
        reason = "height-field symbols exist but public USD authoring is not verified"
    elif not public_dynamic_update_verified:
        state = "update_unverified"
        reason = "public authoring exists but stable runtime update/recook is not verified"
    else:
        state = "available"
        reason = "public authoring and dynamic update probes both passed"
    return HeightFieldCapabilityStatus(
        runtime_label=runtime_label,
        state=state,
        public_schema_symbols=symbols,
        public_authoring_verified=bool(public_authoring_verified),
        public_dynamic_update_verified=bool(public_dynamic_update_verified),
        selected_backend="PhysXHeightFieldContactBackend"
        if state == "available"
        else "TriangleMeshContactBackend",
        reason=reason,
    )


class PhysXHeightFieldContactBackend:
    """Reserved interface marker; construction requires a successful probe."""

    def __init__(self, capability: HeightFieldCapabilityStatus) -> None:
        if not capability.usable:
            raise RuntimeError(
                "[PhysXHeightFieldContactBackend] unavailable: " + capability.reason
            )
        raise NotImplementedError(
            "[PhysXHeightFieldContactBackend] public backend implementation is reserved"
        )

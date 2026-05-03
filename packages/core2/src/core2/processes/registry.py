"""Process registry — factory lookup keyed by `ProcessSpec.name`.

Built-in processes register themselves on package import; downstream code
may register custom processes the same way.
"""

from __future__ import annotations

from collections.abc import Callable

from core2.processes.base import Process, ProcessSpec

_BUILDERS: dict[str, Callable[[ProcessSpec], Process]] = {}


def register(name: str, builder: Callable[[ProcessSpec], Process]) -> None:
    """Register a process builder under `name`.

    Re-registration overwrites silently — useful for tests and for
    swapping implementations during experiments.
    """
    _BUILDERS[name] = builder


def registered_names() -> list[str]:
    return sorted(_BUILDERS)


def build(spec: ProcessSpec) -> Process:
    """Resolve a single spec to a Process instance."""
    try:
        builder = _BUILDERS[spec.name]
    except KeyError:
        known = ", ".join(sorted(_BUILDERS)) or "<none>"
        raise ValueError(
            f"Unknown process {spec.name!r}. Known: {known}"
        ) from None
    return builder(spec)

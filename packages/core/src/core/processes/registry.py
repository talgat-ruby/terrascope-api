"""Process registry — factory lookup keyed by `ProcessSpec.name`.

Built-in processes register themselves on package import; downstream code
may register custom processes the same way.

Opt-in kwargs validation: pass `config_model=` (a Pydantic BaseModel) to
`register()` and the kwargs dict will be validated against it before the
builder is invoked. Builders without a config model see the raw dict —
matching the legacy behavior, so this is fully backwards-compatible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from core.processes.base import Process, ProcessSpec


@dataclass(frozen=True)
class _Entry:
    builder: Callable[[ProcessSpec], Process]
    config_model: type[BaseModel] | None


_BUILDERS: dict[str, _Entry] = {}


def register(
    name: str,
    builder: Callable[[ProcessSpec], Process],
    *,
    config_model: type[BaseModel] | None = None,
) -> None:
    """Register a process builder under `name`.

    Re-registration overwrites silently — useful for tests and for
    swapping implementations during experiments. When `config_model` is
    given, `ProcessSpec.kwargs` is validated against it at `build()` time
    and any unknown / wrongly-typed keys raise a clear `ValueError`.
    """
    _BUILDERS[name] = _Entry(builder=builder, config_model=config_model)


def registered_names() -> list[str]:
    return sorted(_BUILDERS)


def config_model_for(name: str) -> type[BaseModel] | None:
    """Return the Pydantic config model registered for `name`, if any."""
    entry = _BUILDERS.get(name)
    return entry.config_model if entry else None


def build(spec: ProcessSpec) -> Process:
    """Resolve a single spec to a Process instance."""
    try:
        entry = _BUILDERS[spec.name]
    except KeyError:
        known = ", ".join(sorted(_BUILDERS)) or "<none>"
        raise ValueError(
            f"Unknown process {spec.name!r}. Known: {known}"
        ) from None

    if entry.config_model is not None:
        try:
            entry.config_model.model_validate(spec.kwargs)
        except ValidationError as e:
            raise ValueError(
                f"Invalid kwargs for process {spec.name!r}:\n{e}"
            ) from e

    return entry.builder(spec)

"""Tests for the opt-in Pydantic kwargs validation in the process registry.

`msft-buildings` is registered with a `MsftFootprintConfig` model; we use
it to verify that invalid kwargs are rejected at `build()` time, while
processes registered without a config model continue to accept anything.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from core import ProcessSpec, build, register
from core.processes.registry import config_model_for


def test_valid_msft_kwargs_accepted():
    spec = ProcessSpec(
        name="msft-buildings",
        kwargs={"country": "Kazakhstan", "min_area_m2": 5.0},
    )
    proc = build(spec)
    assert proc.name == "msft-buildings"


def test_unknown_kwarg_rejected():
    spec = ProcessSpec(
        name="msft-buildings",
        kwargs={"countryyy": "Kazakhstan"},  # typo
    )
    with pytest.raises(ValueError, match="Invalid kwargs"):
        build(spec)


def test_wrong_type_rejected():
    spec = ProcessSpec(
        name="msft-buildings",
        kwargs={"min_area_m2": "not-a-float"},
    )
    with pytest.raises(ValueError, match="Invalid kwargs"):
        build(spec)


def test_negative_min_area_rejected():
    spec = ProcessSpec(
        name="msft-buildings",
        kwargs={"min_area_m2": -1.0},
    )
    with pytest.raises(ValueError, match="Invalid kwargs"):
        build(spec)


def test_processes_without_config_model_still_accept_any():
    """Backwards-compat: existing processes without a config_model are
    untouched — kwargs are still passed through verbatim.
    """
    class _Stub:
        def __init__(self, spec):
            self.spec = spec
            self.name = spec.name

        def run(self, raster):
            return []

    register("test-stub-no-config", _Stub)
    # Bogus kwargs should pass right through.
    proc = build(
        ProcessSpec(
            name="test-stub-no-config",
            kwargs={"anything": 123, "even_typos": True},
        )
    )
    assert proc.name == "test-stub-no-config"


def test_config_model_for_returns_registered_model():
    model = config_model_for("msft-buildings")
    assert model is not None
    assert issubclass(model, BaseModel)
    assert config_model_for("does-not-exist") is None


def test_register_with_config_model_overrides_existing():
    class _MyCfg(BaseModel):
        model_config = ConfigDict(extra="forbid")
        my_int: int = 0

    class _Stub:
        def __init__(self, spec):
            self.spec = spec
            self.name = spec.name

        def run(self, raster):
            return []

    register("test-stub-config", _Stub, config_model=_MyCfg)
    build(ProcessSpec(name="test-stub-config", kwargs={"my_int": 5}))
    with pytest.raises(ValueError, match="Invalid kwargs"):
        build(ProcessSpec(name="test-stub-config", kwargs={"unknown_key": 1}))

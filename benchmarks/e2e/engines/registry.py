"""The engine registry: the type of an arm's config selects its engine."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from benchmarks.e2e.engines.api import Arm, Engine, EngineConfig
from benchmarks.e2e.engines.megatron_stock.engine import MegatronStockEngine
from benchmarks.e2e.engines.torchtitan.engine import TorchTitanEngine


def _registry(engines: tuple[Engine, ...]) -> Mapping[type[EngineConfig], Engine]:
    """The engines by config type; a repeated config type or name raises."""
    by_type: dict[type[EngineConfig], Engine] = {}
    names: set[str] = set()
    for engine in engines:
        if engine.config_type in by_type:
            raise ValueError(
                f"engines {by_type[engine.config_type].name!r} and "
                f"{engine.name!r} both take {engine.config_type.__name__}"
            )
        if engine.name in names:
            raise ValueError(f"two engines are named {engine.name!r}")
        by_type[engine.config_type] = engine
        names.add(engine.name)
    return MappingProxyType(by_type)


ENGINES: Mapping[type[EngineConfig], Engine] = _registry(
    (TorchTitanEngine(), MegatronStockEngine())
)


def engine_for(arm: Arm) -> Engine:
    """The engine that the type of ``arm.config`` selects."""
    try:
        return ENGINES[type(arm.config)]
    except KeyError as error:
        raise ValueError(
            f"{arm.name}: no engine takes {type(arm.config).__name__}. "
            "Available: "
            + ", ".join(sorted(config.__name__ for config in ENGINES))
        ) from error


def engine_named(name: str) -> Engine:
    """The engine that a manifest records under ``name``."""
    for engine in ENGINES.values():
        if engine.name == name:
            return engine
    raise ValueError(
        f"Unknown engine {name!r}. Available: "
        + ", ".join(sorted(engine.name for engine in ENGINES.values()))
    )

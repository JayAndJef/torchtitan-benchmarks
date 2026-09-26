"""The scenario declaration: the data, the profiler window and the arms.

The module imports the standard library and the standard-library-only
``benchmarks.e2e.engines.api`` alone. ``tests/test_schema.py`` holds that
property.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.e2e.engines.api import Arm, DataSpec, ProfileWindow


@dataclass(frozen=True)
class Scenario:
    """A workload and the arms that measure it.

    ``--seq-len``, ``--batch`` and ``--steps`` override ``data``. A run
    refuses an ``--ac`` value outside ``supported_ac_modes``.
    """

    name: str
    description: str
    data: DataSpec
    window: ProfileWindow
    arms: tuple[Arm, ...]
    supported_ac_modes: tuple[str, ...] = ("sac", "none")

    def arm(self, name: str) -> Arm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ValueError(
            f"Unknown arm {name!r} for scenario {self.name!r}. "
            f"Available arms: {', '.join(arm.name for arm in self.arms)}"
        )

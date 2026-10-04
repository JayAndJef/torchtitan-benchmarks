"""The ``--set <arm>.<field>=<value>`` overrides of an arm's engine config."""

from __future__ import annotations

import dataclasses
import re
import shlex
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from benchmarks.e2e.engines.api import Arm, EngineConfig


_OVERRIDE = re.compile(
    r"(?P<arm>[A-Za-z_][A-Za-z0-9_]*)\.(?P<field>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<op>\+?=)(?P<value>.*)",
    re.DOTALL,
)

_BOOLEANS = {"on": True, "true": True, "off": False, "false": False}


@dataclass(frozen=True)
class Override:
    """One change to one field of one arm's config."""

    arm: str
    field: str
    op: Literal["=", "+="]
    value: str
    """The text after the operator; a ``+=`` value splits with shell rules."""

    def __str__(self) -> str:
        return f"{self.arm}.{self.field}{self.op}{self.value}"


def parse_override(text: str) -> Override:
    """The override that ``<arm>.<field>=<value>`` or ``<arm>.<field>+=<value>`` names."""
    match = _OVERRIDE.fullmatch(text)
    if match is None:
        raise ValueError(
            f"--set {text!r} is not <arm>.<field>=<value> or "
            "<arm>.<field>+=<value>"
        )
    return Override(
        arm=match["arm"], field=match["field"], op=match["op"], value=match["value"]
    )


def field_types(config_type: type[EngineConfig]) -> dict[str, Any]:
    """The resolved annotation of each field of ``config_type``."""
    hints = typing.get_type_hints(config_type)
    return {field.name: hints[field.name] for field in dataclasses.fields(config_type)}


def _converted(override: Override, annotation: Any, current: Any) -> Any:
    """The new value of the field that ``override`` names."""
    where = f"--set {override}"
    origin = typing.get_origin(annotation)
    if origin is tuple:
        if override.op != "+=":
            raise ValueError(
                f"{where}: {override.field} is a list; append to it with +="
            )
        return (*current, *shlex.split(override.value))
    if override.op != "=":
        raise ValueError(
            f"{where}: {override.field} holds one value; set it with ="
        )
    if origin is Literal:
        choices = typing.get_args(annotation)
        if override.value not in choices:
            raise ValueError(
                f"{where}: {override.value!r} is not one of "
                + ", ".join(str(choice) for choice in choices)
            )
        return override.value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        for member in annotation:
            if member.value == override.value:
                return member
        raise ValueError(
            f"{where}: {override.value!r} is not one of "
            + ", ".join(str(member.value) for member in annotation)
        )
    if annotation is bool:
        if override.value not in _BOOLEANS:
            raise ValueError(
                f"{where}: {override.value!r} is not one of "
                + ", ".join(_BOOLEANS)
            )
        return _BOOLEANS[override.value]
    if annotation is int:
        try:
            return int(override.value)
        except ValueError as error:
            raise ValueError(f"{where}: {override.value!r} is not an integer") from error
    if annotation is str:
        return override.value
    raise TypeError(
        f"{where}: no conversion reads a {annotation!r} field; give the field "
        "a Literal, an Enum, a bool, an int, a str or a tuple of str"
    )


def apply_overrides(
    arms: Iterable[Arm], overrides: Iterable[Override]
) -> tuple[Arm, ...]:
    """The arms, with each override applied in order to its arm's config."""
    by_name = {arm.name: arm for arm in arms}
    for override in overrides:
        arm = by_name.get(override.arm)
        if arm is None:
            raise ValueError(
                f"--set {override}: {override.arm!r} is not a selected arm. "
                f"Available: {', '.join(by_name)}"
            )
        types = field_types(type(arm.config))
        if override.field not in types:
            raise ValueError(
                f"--set {override}: {type(arm.config).__name__} has no field "
                f"{override.field!r}. Available: {', '.join(types)}"
            )
        value = _converted(
            override, types[override.field], getattr(arm.config, override.field)
        )
        by_name[arm.name] = dataclasses.replace(
            arm, config=dataclasses.replace(arm.config, **{override.field: value})
        )
    return tuple(by_name.values())

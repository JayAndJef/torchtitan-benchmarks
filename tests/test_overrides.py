"""Tests for the ``--set`` parser and the conversion of each field type."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.engines.api import CompileMode
from benchmarks.e2e.overrides import Override, apply_overrides, parse_override
from benchmarks.e2e.registry import ENGINES


def _applied(*texts: str):
    """The ``engines`` arms by name, after the overrides ``texts``."""
    arms = apply_overrides(
        ENGINES.arms, tuple(parse_override(text) for text in texts)
    )
    return {arm.name: arm for arm in arms}


class ParseTests(unittest.TestCase):
    def test_both_operators_parse(self) -> None:
        self.assertEqual(
            parse_override("megatron_stock.precision=lean"),
            Override("megatron_stock", "precision", "=", "lean"),
        )
        self.assertEqual(
            parse_override("titan_eager.extra_flags+=--a --b=c"),
            Override("titan_eager", "extra_flags", "+=", "--a --b=c"),
        )

    def test_a_value_may_hold_an_equals_sign(self) -> None:
        self.assertEqual(
            parse_override("titan_eager.extra_flags+=--x=1").value, "--x=1"
        )

    def test_a_malformed_text_is_refused(self) -> None:
        for text in ("precision=lean", "megatron_stock.precision", ".x=1", "a.b-c=1"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "is not <arm>.<field>=<value>"):
                    parse_override(text)


class ConversionTests(unittest.TestCase):
    def test_a_literal_takes_one_of_its_values(self) -> None:
        self.assertEqual(
            _applied("megatron_stock.precision=lean")["megatron_stock"].config.precision,
            "lean",
        )
        with self.assertRaisesRegex(ValueError, "'fp8' is not one of stock, lean"):
            _applied("megatron_stock.precision=fp8")

    def test_an_enum_takes_a_member_value(self) -> None:
        self.assertIs(
            _applied("titan_compiled.compile=none")["titan_compiled"].config.compile,
            CompileMode.NONE,
        )
        with self.assertRaisesRegex(ValueError, "'max' is not one of"):
            _applied("titan_compiled.compile=max")

    def test_a_bool_takes_on_off_true_or_false(self) -> None:
        for value, expected in (
            ("on", True),
            ("true", True),
            ("off", False),
            ("false", False),
        ):
            with self.subTest(value=value):
                config = _applied(f"titan_eager.requires_gcc_toolset={value}")[
                    "titan_eager"
                ].config
                self.assertIs(config.requires_gcc_toolset, expected)
        with self.assertRaisesRegex(
            ValueError, "'yes' is not one of on, true, off, false"
        ):
            _applied("titan_eager.requires_gcc_toolset=yes")

    def test_an_int_takes_an_integer(self) -> None:
        arms = _applied("titan_eager.overrides_per_block=3")
        self.assertEqual(arms["titan_eager"].config.overrides_per_block, 3)
        with self.assertRaisesRegex(ValueError, "'three' is not an integer"):
            _applied("titan_eager.overrides_per_block=three")

    def test_a_str_takes_the_text(self) -> None:
        self.assertEqual(
            _applied("titan_eager.config=other")["titan_eager"].config.config, "other"
        )

    def test_a_list_appends_with_shell_rules_in_order(self) -> None:
        config = _applied(
            "megatron_stock.extra_flags+=--moe-permute-fusion",
            "megatron_stock.extra_flags+=--a 'b c'",
        )["megatron_stock"].config
        self.assertEqual(config.extra_flags, ("--moe-permute-fusion", "--a", "b c"))

    def test_a_list_refuses_equals_and_a_scalar_refuses_plus_equals(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "extra_flags is a list; append to it with \\+="
        ):
            _applied("megatron_stock.extra_flags=--x")
        with self.assertRaisesRegex(
            ValueError, "precision holds one value; set it with ="
        ):
            _applied("megatron_stock.precision+=lean")

    def test_an_override_reaches_its_arm_alone(self) -> None:
        arms = _applied("titan_eager.extra_flags+=--x")
        self.assertEqual(arms["titan_eager"].config.extra_flags, ("--x",))
        self.assertEqual(arms["titan_compiled"].config.extra_flags, ())


class RefusalTests(unittest.TestCase):
    def test_an_unselected_arm_is_refused_and_the_choices_are_named(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "'piper' is not a selected arm. Available: titan_compiled, "
            "titan_eager, megatron_stock",
        ):
            _applied("piper.extra_flags+=--x")

    def test_an_unknown_field_is_refused_and_the_choices_are_named(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "MegatronStockConfig has no field 'fusion'. Available: extra_flags, ",
        ):
            _applied("megatron_stock.fusion=on")


if __name__ == "__main__":
    unittest.main()

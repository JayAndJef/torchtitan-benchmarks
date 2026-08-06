"""CPU-only tests for the BENCH_COMPILE_MODE inductor-config hook."""

import contextlib
import importlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper1b.compile_mode import ENV_VAR, apply_compile_mode_from_env


def _apply(mode: str | None) -> str:
    """Run the hook with BENCH_COMPILE_MODE set (or removed) and return stdout."""
    environment = dict(os.environ)
    if mode is None:
        environment.pop(ENV_VAR, None)
    else:
        environment[ENV_VAR] = mode
    stdout = io.StringIO()
    with mock.patch.dict(os.environ, environment, clear=True):
        with contextlib.redirect_stdout(stdout):
            apply_compile_mode_from_env()
    return stdout.getvalue()


class CompileModeTests(unittest.TestCase):
    def setUp(self) -> None:
        from torch._inductor import config as inductor_config

        self.config = inductor_config
        saved = {
            "max_autotune": inductor_config.max_autotune,
            "coordinate_descent_tuning": inductor_config.coordinate_descent_tuning,
        }
        saved_cudagraphs = inductor_config.triton.cudagraphs

        def restore() -> None:
            for key, value in saved.items():
                setattr(inductor_config, key, value)
            inductor_config.triton.cudagraphs = saved_cudagraphs

        self.addCleanup(restore)

    def test_unset_environment_is_a_silent_noop(self) -> None:
        before = (
            self.config.max_autotune,
            self.config.coordinate_descent_tuning,
            self.config.triton.cudagraphs,
        )
        self.assertEqual(_apply(None), "")
        after = (
            self.config.max_autotune,
            self.config.coordinate_descent_tuning,
            self.config.triton.cudagraphs,
        )
        self.assertEqual(before, after)

    def test_default_mode_is_a_silent_noop(self) -> None:
        self.config.triton.cudagraphs = False
        self.assertEqual(_apply("default"), "")
        self.assertFalse(self.config.triton.cudagraphs)

    def test_reduce_overhead_enables_cudagraphs(self) -> None:
        self.config.triton.cudagraphs = False
        self.config.max_autotune = False
        output = _apply("reduce-overhead")
        self.assertTrue(self.config.triton.cudagraphs)
        self.assertFalse(self.config.max_autotune)
        self.assertEqual(
            output.strip(), "[CompileMode] reduce-overhead: triton.cudagraphs=True"
        )

    def test_max_autotune_no_cudagraphs_leaves_cudagraphs_alone(self) -> None:
        self.config.triton.cudagraphs = False
        self.config.max_autotune = False
        self.config.coordinate_descent_tuning = False
        output = _apply("max-autotune-no-cudagraphs")
        self.assertTrue(self.config.max_autotune)
        self.assertTrue(self.config.coordinate_descent_tuning)
        self.assertFalse(self.config.triton.cudagraphs)
        self.assertEqual(
            output.strip(),
            "[CompileMode] max-autotune-no-cudagraphs: "
            "coordinate_descent_tuning=True max_autotune=True",
        )

    def test_max_autotune_applies_the_union(self) -> None:
        self.config.triton.cudagraphs = False
        self.config.max_autotune = False
        self.config.coordinate_descent_tuning = False
        output = _apply("max-autotune")
        self.assertTrue(self.config.max_autotune)
        self.assertTrue(self.config.coordinate_descent_tuning)
        self.assertTrue(self.config.triton.cudagraphs)
        self.assertEqual(
            output.strip(),
            "[CompileMode] max-autotune: coordinate_descent_tuning=True "
            "max_autotune=True triton.cudagraphs=True",
        )

    def test_unknown_mode_raises_without_touching_the_config(self) -> None:
        self.config.max_autotune = False
        with self.assertRaises(RuntimeError):
            _apply("turbo")
        self.assertFalse(self.config.max_autotune)

    def test_config_registry_import_applies_the_mode(self) -> None:
        import piper1b.config_registry

        with mock.patch(
            "piper1b.compile_mode.apply_compile_mode_from_env"
        ) as hook:
            importlib.reload(piper1b.config_registry)
        hook.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

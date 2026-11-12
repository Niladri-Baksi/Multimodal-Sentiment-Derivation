import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_integration_package_is_importable():
    package = importlib.import_module("sentiment_derivation.integration")
    assert package is not None


def test_integration_modules_are_importable():
    expected_modules = [
        "sentiment_derivation.integration",
        "sentiment_derivation.integration.model",
        "sentiment_derivation.integration.pipeline",
    ]

    for module_name in expected_modules:
        module = importlib.import_module(module_name)
        assert module is not None

"""Shared pytest fixtures for the Zaya MLX validation suite.

The model is ~17 GB in BF16. The machine has 36 GB. Two simultaneous copies
will not fit, and pytest does not deduplicate fixtures with the same name
across multiple test files — it treats them as separate fixtures, each
session-scoped to its own file. This file consolidates the model fixture so
the model is loaded **exactly once** per pytest invocation, no matter how
many test files run.

If you find yourself wanting to add a `loaded_model` fixture to an
individual test file: don't. Use this one via the test's argument list.
"""
import pytest


@pytest.fixture(scope="session")
def loaded_model():
    """Load `Zyphra/ZAYA1-8B` exactly once per pytest session."""
    from mlx_lm import load

    model, _tokenizer = load("Zyphra/ZAYA1-8B")
    return model

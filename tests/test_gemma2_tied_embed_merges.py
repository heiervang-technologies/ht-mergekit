"""Regression tests for upstream arcee-ai/mergekit#611.

The reporter believed SLERP/TIES merging Gemma 2 failed because of a library
bug, but the failure was a missing required ``weight`` parameter in their
TIES config (an upstream maintainer confirmed this). These tests pin down
that the merge methods themselves work on Gemma 2's tied-embedding
architecture so that a real regression would surface loudly.
"""

import pytest

from mergekit.config import (
    InputModelDefinition,
    MergeConfiguration,
)
from tests.common import make_picoGemma2, run_and_check_merge


@pytest.fixture(scope="module")
def gemma2_a(tmp_path_factory):
    return make_picoGemma2(tmp_path_factory.mktemp("gemma2_a"))


@pytest.fixture(scope="module")
def gemma2_b(tmp_path_factory):
    return make_picoGemma2(tmp_path_factory.mktemp("gemma2_b"))


def test_gemma2_slerp_merge(gemma2_a, gemma2_b):
    config = MergeConfiguration(
        merge_method="slerp",
        base_model=gemma2_a,
        models=[
            InputModelDefinition(model=gemma2_a),
            InputModelDefinition(model=gemma2_b),
        ],
        parameters={"t": 0.5},
        dtype="bfloat16",
    )
    run_and_check_merge(config)


def test_gemma2_ties_merge_with_weight_and_density(gemma2_a, gemma2_b):
    """TIES requires both ``weight`` and ``density`` per model (the YAML knob
    the upstream maintainer flagged was missing in the original report)."""
    config = MergeConfiguration(
        merge_method="ties",
        base_model=gemma2_a,
        models=[
            InputModelDefinition(
                model=gemma2_a,
                parameters={"weight": 1.0, "density": 0.5},
            ),
            InputModelDefinition(
                model=gemma2_b,
                parameters={"weight": 0.7, "density": 0.5},
            ),
        ],
        dtype="bfloat16",
    )
    run_and_check_merge(config)


def test_gemma2_linear_merge(gemma2_a, gemma2_b):
    """Linear has always worked — keep a test so the three methods are
    compared side-by-side when triaging future reports."""
    config = MergeConfiguration(
        merge_method="linear",
        models=[
            InputModelDefinition(model=gemma2_a, parameters={"weight": 0.5}),
            InputModelDefinition(model=gemma2_b, parameters={"weight": 0.5}),
        ],
        dtype="bfloat16",
    )
    run_and_check_merge(config)

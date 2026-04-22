"""Regression tests for multi-module architectures (e.g. Gemma 3 VL).

Covers upstream bug arcee-ai/mergekit#537 (Gemma 3 VL passthrough crash) and
the accompanying error-message clarity improvement for top-level ``slices:``
against multi-module architectures.
"""

import pytest

from mergekit.architecture import arch_info_for_config
from mergekit.config import (
    InputSliceDefinition,
    MergeConfiguration,
    OutputModuleDefinition,
    OutputSliceDefinition,
)
from mergekit.merge import _model_out_config
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner
from tests.common import make_picoGemma3VL


@pytest.fixture(scope="module")
def gemma3vl_model(tmp_path_factory):
    return make_picoGemma3VL(tmp_path_factory.mktemp("gemma3vl"))


def test_model_out_config_skips_null_num_layers_key(gemma3vl_model):
    """Regression for arcee-ai/mergekit#537.

    The ``multi_modal_projector`` module in gemma3vl.json has no
    ``num_layers_config_key``. When a passthrough-style config touches every
    module (top-level ``slices:``), ``_model_out_config`` must skip modules
    whose key is ``None`` instead of forwarding ``None`` into
    ``set_config_value`` (which would ``AttributeError`` on ``key.split``).
    """
    config = MergeConfiguration(
        merge_method="passthrough",
        slices=[
            OutputSliceDefinition(
                sources=[
                    InputSliceDefinition(
                        model=gemma3vl_model,
                        layer_range=[0, 2],
                    )
                ]
            )
        ],
        dtype="bfloat16",
    )

    # Should not raise; previously crashed with "NoneType has no attribute 'split'".
    cfg_out = _model_out_config(config, _gemma3vl_arch())
    # text_decoder's num_hidden_layers should still be populated.
    assert cfg_out.text_config.num_hidden_layers == 2


def test_slices_on_multi_module_arch_error_names_modules(gemma3vl_model):
    """Top-level ``slices:`` against a multi-module arch must tell the user
    which module keys to wrap their slices under."""
    config = MergeConfiguration(
        merge_method="passthrough",
        slices=[
            OutputSliceDefinition(
                sources=[
                    InputSliceDefinition(
                        model=gemma3vl_model,
                        layer_range=[0, 2],
                    )
                ]
            )
        ],
        dtype="bfloat16",
    )
    planner = MergePlanner(
        config, _gemma3vl_arch(), options=MergeOptions(), out_model_config=None
    )
    with pytest.raises(RuntimeError) as excinfo:
        planner.normalize_config()
    msg = str(excinfo.value)
    # Must name the actual module keys from gemma3vl.json.
    for key in ("text_decoder", "multi_modal_projector", "vision_tower"):
        assert key in msg, f"error message should mention module {key!r}: {msg}"
    # And point at the modules: syntax fix.
    assert "modules:" in msg


def _gemma3vl_arch():
    from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig

    cfg = Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=2,
            head_dim=8,
        ),
        vision_config=SiglipVisionConfig(
            image_size=16,
            patch_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=32,
            intermediate_size=64,
        ),
    )
    cfg.architectures = ["Gemma3ForConditionalGeneration"]
    arch = arch_info_for_config(cfg)
    assert (
        arch is not None
    ), "gemma3vl.json should resolve for Gemma3ForConditionalGeneration"
    return arch

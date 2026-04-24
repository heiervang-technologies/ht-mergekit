"""Regression tests for the ``Missing required parameter`` error message.

Motivated by upstream arcee-ai/mergekit#611 where the reporter's TIES
config omitted the required ``weight`` parameter and the bare error gave
no hint that it was TIES that needed it, let alone where in the YAML to
set it. The improved message names the merge method, the model (for
per-tensor parameters), the tensor, and shows the exact stanza to write.
"""

import pytest

from mergekit.common import ModelReference
from mergekit.config import (
    ConfigReader,
    InputModelDefinition,
    MergeConfiguration,
)


def _reader(merge_method: str, tensor_name: str = None) -> ConfigReader:
    cfg = MergeConfiguration(
        merge_method=merge_method,
        models=[InputModelDefinition(model="stub/model-a")],
    )
    return ConfigReader(config=cfg, t=0.0, tensor_name=tensor_name)


def test_missing_per_tensor_param_mentions_merge_method_and_model():
    reader = _reader("ties", tensor_name="model.embed_tokens.weight")
    model = ModelReference.model_validate("stub/model-a")
    with pytest.raises(RuntimeError) as excinfo:
        reader.parameter("weight", model=model, required=True)
    msg = str(excinfo.value)
    assert "'weight'" in msg
    assert "'ties'" in msg
    assert "stub/model-a" in msg
    assert "model.embed_tokens.weight" in msg
    # Hint must show the per-model YAML stanza.
    assert "models:" in msg
    assert "parameters:" in msg
    assert "weight: <value>" in msg


def test_missing_global_param_points_at_top_level_parameters_block():
    reader = _reader("slerp")
    with pytest.raises(RuntimeError) as excinfo:
        reader.parameter("t", required=True)
    msg = str(excinfo.value)
    assert "'t'" in msg
    assert "'slerp'" in msg
    # Global parameters live at the top level, not under a model.
    assert "top level" in msg
    assert "t: <value>" in msg
    # And crucially NOT under a model.
    assert "- model:" not in msg

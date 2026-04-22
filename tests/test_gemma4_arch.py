"""Static-schema tests for gemma4.json.

Covers google/gemma-4-31B (dense VL) and google/gemma-4-26B-A4B (MoE VL),
which both register as Gemma4ForConditionalGeneration. Does NOT cover the
Elastic (E4B/E2B) variants which ship with QAT calibration tensors, PLE, and
an additional audio_tower module — those are tracked for a follow-up PR, see
``docs/gemma_support.md``.
"""

from transformers import PretrainedConfig

from mergekit.architecture import arch_info_for_config


def _make_config(
    num_hidden_layers: int, vision_num_hidden_layers: int = 27
) -> PretrainedConfig:
    text_config = PretrainedConfig(
        num_hidden_layers=num_hidden_layers,
        vocab_size=262144,
    )
    vision_config = PretrainedConfig(num_hidden_layers=vision_num_hidden_layers)
    cfg = PretrainedConfig(
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=text_config,
        vision_config=vision_config,
    )
    cfg.model_type = "gemma4"
    return cfg


def test_arch_info_resolves_to_gemma4_json():
    arch = arch_info_for_config(_make_config(num_hidden_layers=4))
    assert arch is not None
    assert arch.architectures == ["Gemma4ForConditionalGeneration"]
    assert set(arch.modules.keys()) == {
        "language_model",
        "vision_tower",
        "embed_vision",
    }


def test_language_model_has_dense_and_moe_optional_weights():
    arch = arch_info_for_config(_make_config(num_hidden_layers=2))
    lm = arch.modules["language_model"].architecture
    layer_w = lm.layer_weights(0, _make_config(num_hidden_layers=2))
    names = {w.name for w in layer_w}

    # Always-present (dense core) tensors
    for required in [
        "layers.0.input_layernorm.weight",
        "layers.0.layer_scalar",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.pre_feedforward_layernorm.weight",
        "layers.0.post_feedforward_layernorm.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ]:
        assert required in names, required

    # Shared-V structural omission (full_attention layers in 31B/26B)
    v_proj = next(w for w in layer_w if w.name == "layers.0.self_attn.v_proj.weight")
    assert v_proj.optional, "v_proj must be optional to support shared-V layers"

    # MoE-only tensors must be optional so dense 31B merges don't complain
    for moe_only in [
        "layers.0.pre_feedforward_layernorm_2.weight",
        "layers.0.post_feedforward_layernorm_1.weight",
        "layers.0.post_feedforward_layernorm_2.weight",
        "layers.0.router.proj.weight",
        "layers.0.router.scale",
        "layers.0.router.per_expert_scale",
        "layers.0.experts.gate_up_proj",
        "layers.0.experts.down_proj",
    ]:
        w = next(w for w in layer_w if w.name == moe_only)
        assert w.optional, f"{moe_only} must be optional for dense-variant merges"


def test_lm_head_tied_to_embed_tokens():
    arch = arch_info_for_config(_make_config(num_hidden_layers=2))
    lm = arch.modules["language_model"].architecture
    post = lm.post_weights(_make_config(num_hidden_layers=2))
    lm_head = next(w for w in post if w.name == "lm_head.weight")
    assert lm_head.optional
    assert lm_head.is_embed
    assert "embed_tokens.weight" in (lm_head.tied_names or ())


def test_all_weights_scales_with_num_hidden_layers():
    """Sanity: each `num_hidden_layers` bump adds exactly one layer of
    tensors from the language_model module."""
    arch = arch_info_for_config(_make_config(num_hidden_layers=2))
    cfg2 = _make_config(num_hidden_layers=2)
    cfg3 = _make_config(num_hidden_layers=3)
    n2 = len(arch.all_weights(cfg2))
    n3 = len(arch.all_weights(cfg3))
    per_layer = len(arch.modules["language_model"].architecture.layer_weights(0, cfg2))
    assert n3 - n2 == per_layer


def test_vision_tower_has_expected_encoder_layer_tensors():
    arch = arch_info_for_config(_make_config(num_hidden_layers=2))
    vt = arch.modules["vision_tower"].architecture
    layer_w = vt.layer_weights(0, _make_config(num_hidden_layers=2))
    names = {w.name for w in layer_w}
    for required in [
        "encoder.layers.0.input_layernorm.weight",
        "encoder.layers.0.self_attn.q_proj.linear.weight",
        "encoder.layers.0.self_attn.k_proj.linear.weight",
        "encoder.layers.0.self_attn.v_proj.linear.weight",
        "encoder.layers.0.self_attn.o_proj.linear.weight",
        "encoder.layers.0.mlp.up_proj.linear.weight",
        "encoder.layers.0.mlp.gate_proj.linear.weight",
        "encoder.layers.0.mlp.down_proj.linear.weight",
    ]:
        assert required in names, required


def test_vision_tower_has_std_normalization_post_weights():
    """31B and 26B-A4B ship per-vision-tower std_bias/std_scale tensors."""
    arch = arch_info_for_config(_make_config(num_hidden_layers=2))
    vt = arch.modules["vision_tower"].architecture
    post = vt.post_weights(_make_config(num_hidden_layers=2))
    names = {w.name for w in post}
    assert "std_bias" in names
    assert "std_scale" in names


def test_prefixed_weights_match_checkpoint_naming():
    """ConfiguredModelArchitecture.all_weights() applies per-module weight
    prefixes — the output names must match the 31B/26B-A4B checkpoint
    tensor names (e.g. ``model.language_model.layers.0.input_layernorm.weight``)."""
    from mergekit.architecture.base import ConfiguredModelArchitecture

    cfg = _make_config(num_hidden_layers=2)
    arch = arch_info_for_config(cfg)
    cma = ConfiguredModelArchitecture(info=arch, config=cfg)
    names = {w.name for w in cma.all_weights()}
    for expected in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.1.mlp.down_proj.weight",
        "model.vision_tower.patch_embedder.input_proj.weight",
        "model.vision_tower.patch_embedder.position_embedding_table",
        "model.vision_tower.encoder.layers.0.input_layernorm.weight",
        "model.embed_vision.embedding_projection.weight",
    ]:
        assert expected in names, expected

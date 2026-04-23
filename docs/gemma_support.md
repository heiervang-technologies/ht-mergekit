# Gemma model support in ht-mergekit

This document describes the state of Gemma-family support in the `ht` branch
and what's deferred to follow-up work. For upstream behaviour, see the main
[mergekit README](../README.md#mergekit).

## Supported today

| SKU | Architecture | Modules | Status |
|---|---|---|---|
| `google/gemma`, `google/gemma-*-it` | `GemmaForCausalLM` | single | Upstream |
| `google/gemma-2-*`, `google/gemma-2-*-it` | `Gemma2ForCausalLM` | single (tied lm_head) | Upstream; HT adds SLERP/TIES/linear regression tests (`tests/test_gemma2_tied_embed_merges.py`) |
| `google/gemma-3-*` text | `Gemma3ForCausalLM` | single | Upstream |
| `google/gemma-3-*` multimodal | `Gemma3ForConditionalGeneration` | `text_decoder` + `vision_tower` + `multi_modal_projector` | Upstream + HT regression test for the null-`num_layers_config_key` crash site ([#537](https://github.com/arcee-ai/mergekit/issues/537)) |
| `google/gemma-4-31B`, `google/gemma-4-31B-it` | `Gemma4ForConditionalGeneration` | `language_model` + `vision_tower` + `embed_vision` | HT — dense 60-layer VL |
| `google/gemma-4-26B-A4B`, `google/gemma-4-26B-A4B-it` | `Gemma4ForConditionalGeneration` | same 3-module layout | HT — MoE 30-layer, 128 experts × top-8 |

### Schema trick: one JSON, dense and MoE

`mergekit/_data/architectures/gemma4.json` covers both SKUs. MoE-only
tensors (`router.*`, `experts.*`, the `_1`/`_2` feedforward layernorm
variants) are marked `optional: true` so dense 31B merges don't complain
about missing MoE tensors. Similarly, every sixth decoder layer in 31B and
26B-A4B is a `full_attention` layer that structurally omits `v_proj`
(shared-V: those layers re-use V from the previous `full_attention`
block). Marking `self_attn.v_proj.weight` as `optional: true` works because
`MergePlanner.plan_tensor` checks per-tensor-name presence in each input
checkpoint's index — optional weights absent from the index are silently
skipped for that specific layer without affecting others.

No schema extension was needed. If you're tempted to add one, first check
whether `optional: true` already handles your case the way it handles
shared-V here.

## Not yet supported (follow-up PR)

<!-- TODO(gemma-elastic): land E4B/E2B support in a follow-up PR -->

The Elastic series (`google/gemma-4-E4B`, `google/gemma-4-E4B-it`,
`google/gemma-4-E2B`, `google/gemma-4-E2B-it`) is deliberately out of scope
for the initial Gemma 4 PR. They diverge from the dense/MoE variants on
three axes:

1. **Per-layer embeddings (PLE)**. Top-level tensors
   `embed_tokens_per_layer.weight`, `per_layer_model_projection.weight`,
   `per_layer_projection_norm.weight`, and per-layer
   `per_layer_input_gate.weight`, `per_layer_projection.weight`,
   `post_per_layer_input_norm.weight`. These fit the existing schema
   shape — adding them is a pure JSON extension, no code changes.

2. **Audio tower**. An extra `model.audio_tower.*` module with its own
   encoder layers and an `model.embed_audio.embedding_projection.weight`.
   This pushes the module count from 3 to 5.

3. **QAT calibration tensors**. E-series ships with per-Linear
   `input_max`/`input_min`/`output_max`/`output_min` scalars used for
   INT8 quantization-aware training. Averaging these during a merge
   produces garbage calibration stats — the resulting checkpoint would
   still load but would quantize incorrectly. The follow-up PR needs to
   decide whether to:
   - mark all QAT calibration tensors `optional: true` and document that
     the merged checkpoint must be re-calibrated before inference, or
   - add a `copy_from_base: true` flag on `WeightInfo` that bypasses the
     merge method for that tensor and streams the base model's value
     through (minimal schema extension, requires planner support).

The safe default for now: use `ht-mergekit` for Gemma 4 dense/MoE, and
fall back to upstream + manual fixup for Elastic until the follow-up PR
lands.

## Bug triage notes

### `#537` — Gemma 3 VL passthrough crash

**Root-cause in old mergekit**: `_model_out_config` called
`set_config_value(res, None, ...)` for modules whose JSON has no
`num_layers_config_key`, and `set_config_value` does `key.split('.')`.
`multi_modal_projector` in `gemma3vl.json` is such a module.

**Upstream fix**: already landed in commit
[`83f6b07`](https://github.com/arcee-ai/mergekit/commit/83f6b07) — adds
`if not cfg_key: continue` before the `set_config_value` call.

**HT additions**:

- `tests/test_multi_module_arch.py::test_model_out_config_skips_null_num_layers_key`
  — regression test locking in the upstream fix.
- Improved `plan.py:105` error message for the *follow-up* blocker the
  reporter hit (top-level `slices:` against a multi-module arch): it now
  names the actual module keys and shows the `modules:` stanza to write.

### `#611` — Gemma 2 SLERP/TIES `Missing required parameter weight`

**Root-cause**: the reporter's TIES YAML omitted the required per-model
`weight:` parameter. An upstream maintainer confirmed this on the issue.
SLERP was also called out but SLERP requires `t:` (not `weight:`) — the
reporter's config likely had the same structural mistake.

**HT additions**: `tests/test_gemma2_tied_embed_merges.py` — end-to-end
regressions for SLERP, TIES, and linear on a tiny Gemma 2 with tied
lm_head, so any real future regression surfaces.

### `#632` — Gemma 3 27B SLERP producing 28.8B

Not addressed in this PR. The reporter's config only merges
`text_decoder` but runs a full `Gemma3ForConditionalGeneration` SLERP,
so the output carries unmerged `vision_tower` + `multi_modal_projector`
from the donor. That inflates the parameter count. This is expected
behaviour of per-module merge configs; improving the parameter-count
accounting surfaced to the user is out of scope for the Gemma 4 PR and
tracked for later.

## Known issues / environment

- **`transformers >= 5.5` required.** Under the 5.5+ schema for
  `PretrainedConfig`, pydantic cannot finish building
  `ConfiguredModuleArchitecture`'s schema until `torch` is present in the
  module namespace. `mergekit/architecture/base.py` now imports `torch`
  and calls `model_rebuild()` at the bottom of the file to resolve this
  eagerly — do not remove that block. Older `transformers` (4.x) works
  too; the fix is inert there.
- **`test_lazy_unpickle` fails on `torch >= 2.5`** with
  `'torch.storage.TypedStorage' object has no attribute 'execute'` at
  `mergekit/io/loader.py:61`. This pre-dates Gemma 4 support (observed on
  `ht` branch CI run 24766708225 before this PR). Out of scope for the
  Gemma PR; tracked for a separate follow-up branch (`fix/torch-2.5-lazy-unpickle`).

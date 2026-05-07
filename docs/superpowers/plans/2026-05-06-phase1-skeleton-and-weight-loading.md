# Phase 1: Skeleton + Weight Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully-shaped MLX `nn.Module` hierarchy at `~/code/personal/mlx-lm/mlx_lm/models/zaya.py` that mirrors `Zyphra/transformers @ zaya1`'s `ZayaForCausalLM` exactly at the parameter level. Load all 4 safetensors shards from `Zyphra/ZAYA1-8B` (~16 GB, 2,483 tensors, 8.84 B params) with `strict=True` and verify every weight finds a home with no leftovers.

**Architecture:** Skeleton-first. We define every class with its `nn.Linear`, `nn.Embedding`, `nn.Conv1d`, `nn.RMSNorm`, and parameter shapes — but `__call__` methods can be stubs. The integration test loads weights via `mx.load(...)` plus `model.load_weights(list(weights.items()), strict=True)`. The asymmetry between ATT-only layers (even indices, 13–15 keys each) and MoE-only layers (odd indices, 46–47 keys each, plus optional EDA scale) is built into the model's layer construction. `tie_word_embeddings=True` means `lm_head.weight` is not in the safetensors; we alias it to `embed_tokens.weight` after load.

**Tech Stack:** Python 3.11, MLX, `mlx-lm` (forked at `~/code/personal/mlx-lm`, branch `zaya1`), `safetensors`, pytest. The validation venv is separate from the reference venv (no PyTorch). Editable-install of the mlx-lm fork into the validation venv.

---

## Pre-flight Setup

- [ ] **Confirm Phase 0 state**

Run:
```bash
ls ~/code/personal/zaya1-mlx/reference/activations/smoke/manifest.json && \
test -d ~/code/personal/mlx-lm && \
cd ~/code/personal/mlx-lm && git rev-parse --abbrev-ref HEAD
```
Expected: manifest exists, fork dir exists, on branch `zaya1`.

- [ ] **Confirm safetensors available**

Run:
```bash
ls ~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B/snapshots/*/model-*.safetensors | wc -l
```
Expected: 4 (four shards present).

---

### Task 1: Inventory HF Weight Keys

The dump_activations work already touched this; now we capture the canonical inventory in a doc that both the implementation and `sanitize` reference.

**Files:**
- Create: `~/code/personal/zaya1-mlx/reference/notes/hf-weight-keys.md`

- [ ] **Step 1: Generate the inventory**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -c "
import json, re
from pathlib import Path
from huggingface_hub import snapshot_download
p = Path(snapshot_download('Zyphra/ZAYA1-8B', allow_patterns=['model.safetensors.index.json']))
idx = json.loads((p / 'model.safetensors.index.json').read_text())
wm = idx['weight_map']
print('Total tensors:', len(wm))
print('Total bytes:', idx['metadata']['total_size'])
print('Approx params (bf16):', idx['metadata']['total_size'] // 2)
" 2>&1 | grep -v Fetching
```
Expected output: `Total tensors: 2483`, `Total bytes: 17680978928`, `Approx params (bf16): 8840489464`.

- [ ] **Step 2: Write the inventory doc**

Write `reference/notes/hf-weight-keys.md`:
```markdown
# HF safetensors weight key inventory: Zyphra/ZAYA1-8B

**Source:** `model.safetensors.index.json` from the HF snapshot.
**Total tensors:** 2,483
**Total bf16 bytes:** 17,680,978,928 (~16.5 GiB)
**Approx params (bf16):** 8,840,489,464 (~8.84 B)

## Top-level keys (6 total)

- `model.embed_tokens.weight` — embedding table, shape (vocab_size=262272, hidden_size=2048)
- `model.final_norm.weight` — final RMSNorm gain, shape (hidden_size=2048,)
- `model.res_scale.hidden_states_bias` — final ResidualScaling, shape (hidden_size,)
- `model.res_scale.hidden_states_scale` — final ResidualScaling, shape (hidden_size,)
- `model.res_scale.residual_bias` — final ResidualScaling, shape (hidden_size,)
- `model.res_scale.residual_scale` — final ResidualScaling, shape (hidden_size,)

**`lm_head.weight` is NOT in the safetensors** — the model uses `tie_word_embeddings=True`, so lm_head is aliased to embed_tokens at load time.

## Per-layer keys

### Even layers (0, 2, 4, …, 78) — ATT layer

Layer 0 has 13 keys (no `res_scale.residual_*` because it's the first layer).
Other even layers have 15 keys.

```
model.layers.{i}.input_norm.weight
model.layers.{i}.res_scale.hidden_states_bias
model.layers.{i}.res_scale.hidden_states_scale
model.layers.{i}.res_scale.residual_bias        # only if i != 0
model.layers.{i}.res_scale.residual_scale       # only if i != 0
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.self_attn.qkv.linear_q.weight
model.layers.{i}.self_attn.qkv.linear_k.weight
model.layers.{i}.self_attn.qkv.val_proj1.weight
model.layers.{i}.self_attn.qkv.val_proj2.weight
model.layers.{i}.self_attn.qkv.conv_qk.0.weight
model.layers.{i}.self_attn.qkv.conv_qk.0.bias
model.layers.{i}.self_attn.qkv.conv_qk.1.weight
model.layers.{i}.self_attn.qkv.conv_qk.1.bias
model.layers.{i}.self_attn.qkv.temp
```

### Odd layers (1, 3, 5, …, 79) — MoE layer

Layer 1 has 46 keys (no `router_states_scale` because EDA is gated off for the first MoE layer).
Other odd layers have 47 keys.

```
model.layers.{i}.input_norm.weight
model.layers.{i}.res_scale.hidden_states_bias
model.layers.{i}.res_scale.hidden_states_scale
model.layers.{i}.res_scale.residual_bias        # always present for odd layers
model.layers.{i}.res_scale.residual_scale
model.layers.{i}.zaya_block.router.balancing_biases     # buffer, not parameter
model.layers.{i}.zaya_block.router.down_proj.weight
model.layers.{i}.zaya_block.router.down_proj.bias
model.layers.{i}.zaya_block.router.rmsnorm_eda.weight
model.layers.{i}.zaya_block.router.router_states_scale  # only if i != 1 (EDA)
model.layers.{i}.zaya_block.router.router_mlp.0.weight
model.layers.{i}.zaya_block.router.router_mlp.0.bias
model.layers.{i}.zaya_block.router.router_mlp.2.weight
model.layers.{i}.zaya_block.router.router_mlp.2.bias
model.layers.{i}.zaya_block.router.router_mlp.4.weight  # NB: no bias on the output linear
model.layers.{i}.zaya_block.experts.local_experts.{e}.linear_fc1.weight  # for e in 0..15
model.layers.{i}.zaya_block.experts.local_experts.{e}.linear_fc2.weight  # for e in 0..15
```

## Shape reference

| key | shape |
|---|---|
| embed_tokens.weight | (262272, 2048) |
| final_norm.weight, res_scale.* | (2048,) |
| input_norm.weight | (2048,) |
| res_scale.hidden_states_scale | (2048,) — diagonal affine, per-feature |
| self_attn.o_proj.weight | (2048, 1024) — input dim is hidden_size//2 (CCA query compression) |
| self_attn.qkv.linear_q.weight | (1024, 2048) — 8 q heads × 128 dim |
| self_attn.qkv.linear_k.weight | (256, 2048) — 2 kv heads × 128 dim |
| self_attn.qkv.val_proj1.weight | (128, 2048) — latent_k_dim/2 |
| self_attn.qkv.val_proj2.weight | (128, 2048) |
| self_attn.qkv.conv_qk.0.weight | (1280, 1, 2) — depthwise (groups=1280), kernel=2 |
| self_attn.qkv.conv_qk.0.bias | (1280,) |
| self_attn.qkv.conv_qk.1.weight | (1280, 128, 2) — grouped (groups=10), kernel=2; in/out per group = 128 |
| self_attn.qkv.conv_qk.1.bias | (1280,) |
| self_attn.qkv.temp | (2,) — per-KV-head temperature |
| zaya_block.router.balancing_biases | (17,) — buffer, num_experts+1 (16 real + 1 skip) |
| zaya_block.router.down_proj.weight | (256, 2048) |
| zaya_block.router.down_proj.bias | (256,) |
| zaya_block.router.rmsnorm_eda.weight | (256,) |
| zaya_block.router.router_states_scale | (256,) — EDA per-feature gain |
| zaya_block.router.router_mlp.0.weight | (256, 256) |
| zaya_block.router.router_mlp.0.bias | (256,) |
| zaya_block.router.router_mlp.2.weight | (256, 256) |
| zaya_block.router.router_mlp.2.bias | (256,) |
| zaya_block.router.router_mlp.4.weight | (17, 256) — output is num_experts+1 |
| zaya_block.experts.local_experts.{e}.linear_fc1.weight | (4096, 2048) — ffn_hidden_size |
| zaya_block.experts.local_experts.{e}.linear_fc2.weight | (2048, 2048) — ffn_hidden_size_out = ffn/2 due to gated linear unit |

Conv1d weight conventions: PyTorch's `Conv1d(in_ch, out_ch, kernel, groups=g)` weight shape is `(out_ch, in_ch/groups, kernel)`. For `groups=in_ch=out_ch=1280` (depthwise), in_ch/groups = 1. For `groups=10` (grouped), in_ch/groups = 1280/10 = 128.
```

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/notes/hf-weight-keys.md
git commit -m "Phase 1 task 1: HF safetensors weight key inventory"
```

---

### Task 2: Document mlx-lm Conventions

Read existing mlx-lm models to learn conventions before writing zaya.py.

**Files:**
- Create: `~/code/personal/zaya1-mlx/reference/notes/mlx-lm-conventions.md`

- [ ] **Step 1: Read three reference models**

Read in this order, looking for: ModelArgs structure, Model class structure, sanitize signature, how `__call__` is shaped, weight tying handling, MoE patterns:
1. `~/code/personal/mlx-lm/mlx_lm/models/base.py`
2. `~/code/personal/mlx-lm/mlx_lm/models/llama.py` (simplest, has tie_word_embeddings)
3. `~/code/personal/mlx-lm/mlx_lm/models/jamba.py` (has MoE; closest analogue)

- [ ] **Step 2: Write the conventions doc**

Write `reference/notes/mlx-lm-conventions.md` with at minimum:
```markdown
# mlx-lm conventions for new model ports

Based on reading: `mlx_lm/models/{base.py, llama.py, jamba.py}`.

## File structure

A model lives in a single file `mlx_lm/models/<name>.py` with these top-level definitions:

- `@dataclass class ModelArgs(BaseModelArgs):` — config from `config.json`. Field names match the JSON keys exactly. `BaseModelArgs` from `mlx_lm.models.base`.
- N module classes (Attention, MLP, etc.) — each `class X(nn.Module)` with `__init__(self, args: ModelArgs)` and `__call__(...)`.
- `class <ModelName>(nn.Module):` — the embedding + layers + final norm wrapper. Conventionally named after the model (e.g., `Mistral`, `Llama`).
- `class Model(nn.Module):` — the canonical mlx-lm name; wraps the inner model + lm_head; this is what `mlx_lm.load` instantiates. **Always named `Model`**, not `ZayaForCausalLM`.

## Required Model class members

- `args` (the ModelArgs)
- `model_type` (string, must match the config.json `model_type` field)
- `model` (the inner model with embedding + layers + final_norm)
- `lm_head` (Linear, optional if tied)
- `__call__(self, inputs: mx.array, cache=None) -> mx.array` returning logits
- `sanitize(self, weights: dict) -> dict` — remap HF weight keys to MLX names; called by mlx_lm.load
- `layers` property (returns the decoder layer list)
- optional `head_dim`, `n_kv_heads` for KV cache shape

## Weight tying for tie_word_embeddings

Two patterns observed:
1. Llama-style: `lm_head` is `None` if tied; `__call__` does `out = self.model.embed_tokens.as_linear(out)` instead of `lm_head(out)`.
2. Jamba-style: `lm_head` is always a Linear; `sanitize` aliases `embed_tokens.weight` → `lm_head.weight` if tied.

**For Zaya**: use Llama-style (don't define `lm_head` when tied; use `as_linear`). It avoids carrying duplicate parameters and matches what HF actually shipped (no `lm_head.weight` in safetensors).

## sanitize patterns

`sanitize` runs before `model.load_weights`. Common transforms:
- Drop unused HF keys (e.g., `lm_head.weight` if tied + we use as_linear).
- Rename from HF to MLX naming if any divergence (often none for new ports — match HF naming exactly).
- Reshape Conv1d weights if MLX expects different layout.

For Zaya: HF and MLX naming should match 1:1 (we choose MLX names that mirror HF). Sanitize is mostly a passthrough; its main job is to log unmapped keys for diagnostics.

## Conv1d layout

PyTorch nn.Conv1d weight: (out_channels, in_channels/groups, kernel_size).
MLX nn.Conv1d weight: (out_channels, kernel_size, in_channels/groups) — kernel and in_ch are swapped.

Our `sanitize` must transpose conv_qk weights from PyTorch (out, in/g, k) to MLX (out, k, in/g) layout.
```

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/notes/mlx-lm-conventions.md
git commit -m "Phase 1 task 2: document mlx-lm conventions for new model ports"
```

---

### Task 3: Validation venv with editable mlx-lm

The validation venv is separate from `reference/.venv` (which has PyTorch and is large). It contains MLX, the editable mlx-lm fork, pytest, numpy.

**Files:**
- Create: `~/code/personal/zaya1-mlx/validation/pyproject.toml`
- Create: `~/code/personal/zaya1-mlx/validation/.python-version`
- Create: `~/code/personal/zaya1-mlx/validation/.gitignore`

- [ ] **Step 1: Write `validation/pyproject.toml`**

```toml
[project]
name = "zaya1-mlx-validation"
version = "0.0.0"
description = "MLX + editable mlx-lm fork for layer-by-layer parity validation against PyTorch reference dumps."
requires-python = ">=3.11,<3.13"

dependencies = [
  "mlx>=0.20",
  "numpy>=1.26,<2.0",
  "safetensors>=0.4",
  "pytest>=8.0",
]
```

- [ ] **Step 2: Write `validation/.python-version`**

Contents: `3.11`

- [ ] **Step 3: Write `validation/.gitignore`**

Contents:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Create the venv and install**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python "mlx>=0.20" "numpy>=1.26,<2.0" "safetensors>=0.4" "pytest>=8.0"
uv pip install --python .venv/bin/python -e ~/code/personal/mlx-lm
```
Expected: install completes; mlx-lm is editable.

- [ ] **Step 5: Smoke-test MLX**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -c "import mlx.core as mx; import mlx_lm; print('mlx', mx.__version__); print('mlx_lm path:', mlx_lm.__file__)"
```
Expected: prints mlx version, and mlx_lm path that points into `~/code/personal/mlx-lm/mlx_lm/...` (NOT site-packages — confirming editable install).

- [ ] **Step 6: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add -f validation/pyproject.toml validation/.python-version validation/.gitignore
git commit -m "Phase 1 task 3: validation venv with editable mlx-lm + MLX deps"
```

---

### Task 4: Failing weight-loading test

This is the gate test that drives all subsequent skeleton work. It loads weights, calls `mlx_lm.load("Zyphra/ZAYA1-8B")`, asserts no missing or extra keys, and asserts param count.

**Files:**
- Create: `~/code/personal/zaya1-mlx/validation/test_weight_loading.py`

- [ ] **Step 1: Write the test**

Write `validation/test_weight_loading.py`:
```python
"""Phase 1 gate test: load every Zaya safetensor weight into the MLX skeleton.

The skeleton's __call__ methods may be stubs; only the parameter shapes need
to match. Test passes when:
  - mlx_lm.load("Zyphra/ZAYA1-8B") completes without error
  - All 2,483 HF tensors map to skeleton params (no missing, no leftovers)
  - tie_word_embeddings is honored (no separate lm_head.weight loaded)
  - Parameter count matches the HF total (~8.84B)
"""
import mlx.core as mx
import mlx.nn as nn
import pytest


EXPECTED_HF_TENSORS = 2483
# 8.84B params + lm_head weight aliased from embed_tokens (already counted in HF total)
EXPECTED_PARAMS_LOWER = 8_800_000_000
EXPECTED_PARAMS_UPPER = 8_900_000_000


def _count_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    total = 0
    for _, v in tree_flatten(model.parameters()):
        if isinstance(v, mx.array):
            total += v.size
    return total


@pytest.fixture(scope="session")
def loaded_model():
    from mlx_lm import load

    model, _tokenizer = load("Zyphra/ZAYA1-8B")
    return model


def test_model_type_is_zaya(loaded_model):
    assert loaded_model.model_type == "zaya"


def test_layer_count(loaded_model):
    assert len(loaded_model.layers) == 80


def test_layers_alternate_att_moe(loaded_model):
    """Even indices: ATT layers (have self_attn). Odd: MoE (have zaya_block)."""
    for i, layer in enumerate(loaded_model.layers):
        if i % 2 == 0:
            assert hasattr(layer, "self_attn"), f"layer {i} should be ATT (have self_attn)"
            assert not hasattr(layer, "zaya_block"), f"layer {i} should be ATT (no zaya_block)"
        else:
            assert hasattr(layer, "zaya_block"), f"layer {i} should be MoE (have zaya_block)"
            assert not hasattr(layer, "self_attn"), f"layer {i} should be MoE (no self_attn)"


def test_total_param_count(loaded_model):
    n = _count_params(loaded_model)
    assert EXPECTED_PARAMS_LOWER <= n <= EXPECTED_PARAMS_UPPER, (
        f"Expected ~8.84B params, got {n:,}"
    )


def test_embed_and_lm_head_share_weights(loaded_model):
    """tie_word_embeddings: lm_head should be None and as_linear used."""
    assert getattr(loaded_model, "lm_head", None) is None
    embed = loaded_model.model.embed_tokens
    assert embed.weight.shape == (262272, 2048)


def test_layer_0_att_shapes(loaded_model):
    """Spot-check CCA shapes on layer 0."""
    cca = loaded_model.layers[0].self_attn.qkv
    assert cca.linear_q.weight.shape == (1024, 2048)
    assert cca.linear_k.weight.shape == (256, 2048)
    assert cca.val_proj1.weight.shape == (128, 2048)
    assert cca.val_proj2.weight.shape == (128, 2048)
    # Conv1d in MLX: weight shape (out_channels, kernel, in_channels/groups)
    assert cca.conv_qk.layers[0].weight.shape == (1280, 2, 1)  # depthwise: in/groups=1
    assert cca.conv_qk.layers[1].weight.shape == (1280, 2, 128)  # grouped 10: in/groups=128
    assert cca.temp.shape == (2,)
    o_proj_weight = loaded_model.layers[0].self_attn.o_proj.weight
    assert o_proj_weight.shape == (2048, 1024)


def test_layer_1_moe_shapes_and_no_eda(loaded_model):
    """Spot-check router on layer 1; verify EDA scale absent (zaya_first_layer=1)."""
    router = loaded_model.layers[1].zaya_block.router
    assert router.down_proj.weight.shape == (256, 2048)
    assert router.down_proj.bias.shape == (256,)
    assert router.balancing_biases.shape == (17,)
    # Layer 1 is the first MoE layer, no EDA scale
    assert getattr(router, "router_states_scale", None) is None
    # 16 real experts (skip is handled by MoD in code, not as a separate expert weight)
    experts = loaded_model.layers[1].zaya_block.experts.local_experts
    assert len(experts) == 16
    assert experts[0].linear_fc1.weight.shape == (4096, 2048)
    assert experts[0].linear_fc2.weight.shape == (2048, 2048)


def test_layer_3_moe_has_eda(loaded_model):
    """Layer 3 is a non-first MoE layer; EDA scale must exist."""
    router = loaded_model.layers[3].zaya_block.router
    assert router.router_states_scale.shape == (256,)
```

- [ ] **Step 2: Run the test, verify it fails**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_weight_loading.py -v
```
Expected: fails because `model_type="zaya"` is not registered in `mlx-lm/mlx_lm/models/__init__.py`. Error message will mention unknown model type or missing module.

- [ ] **Step 3: Commit the test**

```bash
cd ~/code/personal/zaya1-mlx
git add validation/test_weight_loading.py
git commit -m "Phase 1 task 4: failing weight-loading integration test"
```

---

### Task 5: ModelArgs dataclass

**Files:**
- Create: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

- [ ] **Step 1: Write `ModelArgs`**

Write the start of `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`:
```python
# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "zaya"
    hidden_size: int = 2048
    num_hidden_layers: int = 80
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    num_query_groups: int = 2
    cca_num_q_heads: int = 8
    cca_time0: int = 2
    cca_time1: int = 2
    ffn_hidden_size: int = 4096
    num_experts: int = 16
    moe_router_topk: int = 1
    zaya_mlp_expansion: int = 256
    zaya_use_mod: bool = True
    zaya_use_eda: bool = True
    vocab_size: int = 262272
    max_position_embeddings: int = 131072
    partial_rotary_factor: float = 0.5
    rope_theta: float = 5000000.0
    rope_scaling: Optional[dict] = None
    norm_epsilon: float = 1e-5
    attention_bias: bool = False
    lm_head_bias: bool = False
    add_bias_linear: bool = False
    tie_word_embeddings: bool = True
    residual_in_fp32: bool = True
    scale_residual_merge: bool = True
    activation_func: str = "swiglu"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
```

`BaseModelArgs` is the mlx-lm convention; check `~/code/personal/mlx-lm/mlx_lm/models/base.py` for what it provides (typically `from_dict` and field aliasing).

- [ ] **Step 2: Confirm imports work**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -c "from mlx_lm.models.zaya import ModelArgs; a = ModelArgs(); print(a.head_dim, a.tie_word_embeddings)"
```
Expected: `128 True`.

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 1 task 5: ZayaConfig → MLX ModelArgs dataclass"
```

---

### Task 6: Foundational module classes

ResidualScaling, MLP, SequentialMLP. These are simple and have no external dependencies inside the model.

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

- [ ] **Step 1: Append `ResidualScaling`**

Append to `mlx_lm/models/zaya.py`:
```python
class ResidualScaling(nn.Module):
    """Per-feature affine on residual streams before merging.

    Per modular_zaya.py:1003-1033. Layer 0 only has hidden_states_*
    parameters; non-first layers also have residual_*.
    """

    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.not_first_layer = layer_n != 0
        self.hidden_states_scale = mx.ones((args.hidden_size,))
        self.hidden_states_bias = mx.zeros((args.hidden_size,))
        if self.not_first_layer:
            self.residual_scale = mx.ones((args.hidden_size,))
            self.residual_bias = mx.zeros((args.hidden_size,))

    def __call__(self, residual, hidden_states):
        hidden_states = (hidden_states + self.hidden_states_bias) * self.hidden_states_scale
        if self.not_first_layer:
            residual = (residual + self.residual_bias) * self.residual_scale
        return residual, hidden_states
```

- [ ] **Step 2: Append `MLP` (single SwiGLU expert)**

Append:
```python
class MLP(nn.Module):
    """Single SwiGLU expert. Per modular_zaya.py:1190-1275.

    Note: ffn_hidden_size_out is ffn_hidden_size // 2 due to gated linear unit.
    add_bias_linear is False per config, so no bias on either Linear.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        ffn_out = args.ffn_hidden_size // 2  # gated linear unit halves output
        self.linear_fc1 = nn.Linear(args.hidden_size, args.ffn_hidden_size, bias=args.add_bias_linear)
        self.linear_fc2 = nn.Linear(ffn_out, args.hidden_size, bias=args.add_bias_linear)

    def __call__(self, x):
        h = self.linear_fc1(x)
        a, b = mx.split(h, 2, axis=-1)
        return self.linear_fc2(nn.silu(a) * b)
```

- [ ] **Step 3: Append `SequentialMLP`**

Append:
```python
class SequentialMLP(nn.Module):
    """Container of MoE experts. Per modular_zaya.py:1278-1326.

    Holds a list of MLP modules. Forward routing logic is implemented later
    (in ZayaBlock) — at the skeleton stage we just declare the parameters.
    """

    def __init__(self, args: ModelArgs, num_local_experts: int):
        super().__init__()
        self.local_experts = [MLP(args) for _ in range(num_local_experts)]
```

- [ ] **Step 4: Smoke-test instantiation**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -c "
from mlx_lm.models.zaya import ModelArgs, ResidualScaling, MLP, SequentialMLP
a = ModelArgs()
rs = ResidualScaling(a, layer_n=0)
print('rs first layer params:', list(rs.parameters().keys()))
rs = ResidualScaling(a, layer_n=1)
print('rs non-first params:', list(rs.parameters().keys()))
mlp = MLP(a)
print('mlp linear_fc1.weight shape:', mlp.linear_fc1.weight.shape)
print('mlp linear_fc2.weight shape:', mlp.linear_fc2.weight.shape)
seq = SequentialMLP(a, 16)
print('seq.local_experts len:', len(seq.local_experts))
"
```
Expected:
```
rs first layer params: ['hidden_states_scale', 'hidden_states_bias']
rs non-first params: ['hidden_states_scale', 'hidden_states_bias', 'residual_scale', 'residual_bias']
mlp linear_fc1.weight shape: (4096, 2048)
mlp linear_fc2.weight shape: (2048, 2048)
seq.local_experts len: 16
```

- [ ] **Step 5: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 1 task 6: ResidualScaling, MLP, SequentialMLP"
```

---

### Task 7: CCA + ZayaAttention

CCA is the most novel attention component. The Conv1d layout difference between PyTorch and MLX matters here.

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

- [ ] **Step 1: Append `CCA`**

Append:
```python
class CCA(nn.Module):
    """Compressed Causal Attention. Per modular_zaya.py:285-521.

    Replaces standard QKV with: linear projections + two-stage depthwise 1D
    causal conv on concatenated Q+K + L2-normalized Q/K with per-KV-head temp +
    two-stream V (current + time-shifted hidden state).

    Skeleton only — forward will be implemented in Phase 3.
    """

    def __init__(self, args: ModelArgs, layer_number: int):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_size = args.hidden_size
        self.num_kv_heads = args.num_query_groups  # 2
        self.num_q_heads = args.cca_num_q_heads  # 8
        self.num_heads = args.num_attention_heads  # 16 (for head_dim calc)
        self.head_dim = args.hidden_size // self.num_heads  # 128
        self.latent_k_dim = self.num_kv_heads * self.head_dim  # 256
        self.latent_q_dim = self.num_q_heads * self.head_dim  # 1024
        self.cca_time0 = args.cca_time0
        self.cca_time1 = args.cca_time1

        self.linear_q = nn.Linear(self.hidden_size, self.latent_q_dim, bias=args.attention_bias)
        self.linear_k = nn.Linear(self.hidden_size, self.latent_k_dim, bias=args.attention_bias)
        self.val_proj1 = nn.Linear(self.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias)
        self.val_proj2 = nn.Linear(self.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias)

        in_out_ch = self.latent_k_dim + self.latent_q_dim  # 1280
        self.conv_qk = nn.Sequential(
            nn.Conv1d(
                in_channels=in_out_ch,
                out_channels=in_out_ch,
                kernel_size=self.cca_time0,
                groups=in_out_ch,  # depthwise: groups=in_ch=out_ch
                padding=0,
                stride=1,
                bias=True,
            ),
            nn.Conv1d(
                in_channels=in_out_ch,
                out_channels=in_out_ch,
                kernel_size=self.cca_time1,
                groups=(self.num_kv_heads + self.num_q_heads),  # 10
                padding=0,
                stride=1,
                bias=True,
            ),
        )
        # Per-KV-head temperature
        self.temp = mx.zeros((self.num_kv_heads,))


class ZayaAttention(nn.Module):
    """Wraps CCA + standard scaled dot product attention.

    Per modular_zaya.py:524-656. Skeleton only — forward implemented in Phase 4.
    """

    def __init__(self, args: ModelArgs, layer_number: int):
        super().__init__()
        self.qkv = CCA(args, layer_number)
        # o_proj input dim is hidden_size // 2 because CCA produces only 8
        # effective query heads (cca_num_q_heads), so the post-attn flat dim is
        # 8 * head_dim = 1024 = hidden_size // 2.
        self.o_proj = nn.Linear(
            args.hidden_size // 2,
            args.hidden_size,
            bias=args.attention_bias,
        )
```

- [ ] **Step 2: Smoke-test**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -c "
from mlx_lm.models.zaya import ModelArgs, CCA, ZayaAttention
a = ModelArgs()
cca = CCA(a, layer_number=0)
print('linear_q:', cca.linear_q.weight.shape)
print('linear_k:', cca.linear_k.weight.shape)
print('val_proj1:', cca.val_proj1.weight.shape)
print('val_proj2:', cca.val_proj2.weight.shape)
print('conv_qk[0]:', cca.conv_qk.layers[0].weight.shape, 'bias:', cca.conv_qk.layers[0].bias.shape)
print('conv_qk[1]:', cca.conv_qk.layers[1].weight.shape, 'bias:', cca.conv_qk.layers[1].bias.shape)
print('temp:', cca.temp.shape)
attn = ZayaAttention(a, layer_number=0)
print('o_proj:', attn.o_proj.weight.shape)
"
```
Expected:
```
linear_q: (1024, 2048)
linear_k: (256, 2048)
val_proj1: (128, 2048)
val_proj2: (128, 2048)
conv_qk[0]: (1280, 2, 1) bias: (1280,)
conv_qk[1]: (1280, 2, 128) bias: (1280,)
temp: (2,)
o_proj: (2048, 1024)
```

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 1 task 7: CCA + ZayaAttention skeleton"
```

---

### Task 8: ZayaRouter + ZayaBlock

Router has the most conditional structure: `router_states_scale` only for non-first MoE layers; `balancing_biases` is a buffer (mlx pattern: `mx.array` member without making it a parameter).

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

- [ ] **Step 1: Append `ZayaRouter`**

Append:
```python
ZAYA_FIRST_MOE_LAYER = 1  # hardcoded in modular_zaya.py:1089


class ZayaRouter(nn.Module):
    """MoE router with optional EDA. Per modular_zaya.py:1036-1187.

    Skeleton only — forward implemented in Phase 6.

    EDA gate: enabled when `args.zaya_use_eda` is True AND layer_number != 1
    (the first MoE layer in the global index). The first MoE layer skips EDA.
    """

    def __init__(self, args: ModelArgs, layer_number: int):
        super().__init__()
        self.layer_number = layer_number
        self.use_mod = args.zaya_use_mod
        # num_experts includes a skip expert when MoD is on
        self.num_experts = args.num_experts + 1 if self.use_mod else args.num_experts
        self.mlp_expansion = args.zaya_mlp_expansion

        self.down_proj = nn.Linear(args.hidden_size, self.mlp_expansion, bias=True)

        self.use_eda = args.zaya_use_eda and (layer_number != ZAYA_FIRST_MOE_LAYER)

        self.rmsnorm_eda = nn.RMSNorm(self.mlp_expansion, eps=args.norm_epsilon)
        if self.use_eda:
            self.router_states_scale = mx.ones((self.mlp_expansion,))

        # Three-layer MLP: D -> D -> D -> num_experts (with GELU between)
        self.router_mlp = nn.Sequential(
            nn.Linear(self.mlp_expansion, self.mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(self.mlp_expansion, self.mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(self.mlp_expansion, self.num_experts, bias=False),
        )

        # balancing_biases is loaded from the safetensors. Init values matter
        # only as defaults if no checkpoint is loaded: zeros, with the skip
        # expert at -1.0 when MoD is on.
        if self.use_mod:
            init_bb = [0.0] * (self.num_experts - 1) + [-1.0]
        else:
            init_bb = [0.0] * self.num_experts
        self.balancing_biases = mx.array(init_bb)


class ZayaBlock(nn.Module):
    """MoE block: router + experts + MoD skip. Per modular_zaya.py:1329-1422.

    Skeleton only — forward implemented in Phase 6.
    """

    def __init__(self, args: ModelArgs, layer_number: int):
        super().__init__()
        self.use_mod = args.zaya_use_mod
        self.router = ZayaRouter(args, layer_number)
        # SequentialMLP holds num_experts MLPs (the skip-expert is handled by
        # passing tokens through unchanged in code; not a real MLP).
        self.experts = SequentialMLP(args, args.num_experts)
```

- [ ] **Step 2: Smoke-test**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -c "
from mlx_lm.models.zaya import ModelArgs, ZayaRouter, ZayaBlock
a = ModelArgs()
r1 = ZayaRouter(a, layer_number=1)  # first MoE layer, no EDA
r3 = ZayaRouter(a, layer_number=3)  # non-first MoE, has EDA
print('r1.use_eda:', r1.use_eda, 'r3.use_eda:', r3.use_eda)
print('r1 has router_states_scale:', hasattr(r1, 'router_states_scale'))
print('r3 has router_states_scale:', hasattr(r3, 'router_states_scale'))
print('r1.balancing_biases:', r1.balancing_biases.shape, 'last value:', r1.balancing_biases[-1].item())
print('r3.down_proj.weight:', r3.down_proj.weight.shape)
print('r3.router_mlp[4].weight:', r3.router_mlp.layers[4].weight.shape)
b = ZayaBlock(a, layer_number=3)
print('b.experts.local_experts len:', len(b.experts.local_experts))
"
```
Expected:
```
r1.use_eda: False r3.use_eda: True
r1 has router_states_scale: False
r3 has router_states_scale: True
r1.balancing_biases: (17,) last value: -1.0
r3.down_proj.weight: (256, 2048)
r3.router_mlp[4].weight: (17, 256)
b.experts.local_experts len: 16
```

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 1 task 8: ZayaRouter (with conditional EDA + balancing biases) + ZayaBlock"
```

---

### Task 9: Decoder layer classes

The two layer flavors. Each holds an input_norm, an optional res_scale, and either ZayaAttention or ZayaBlock.

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

- [ ] **Step 1: Append `ZayaDecoderATTLayer` and `ZayaDecoderMLPLayer`**

Append:
```python
class ZayaDecoderATTLayer(nn.Module):
    """Even-indexed decoder layer (CCA self-attention).

    Per modular_zaya.py:909-1000. Skeleton only.
    """

    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.layer_n = layer_n
        self.self_attn = ZayaAttention(args, layer_n)
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)


class ZayaDecoderMLPLayer(nn.Module):
    """Odd-indexed decoder layer (MoE).

    Per modular_zaya.py:1425-1533. Skeleton only.
    """

    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.layer_n = layer_n
        self.zaya_block = ZayaBlock(args, layer_n)
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, layer_n)
```

- [ ] **Step 2: Smoke-test**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -c "
from mlx_lm.models.zaya import ModelArgs, ZayaDecoderATTLayer, ZayaDecoderMLPLayer
a = ModelArgs()
att = ZayaDecoderATTLayer(a, layer_n=0)
moe = ZayaDecoderMLPLayer(a, layer_n=1)
print('att has self_attn:', hasattr(att, 'self_attn'))
print('att has zaya_block:', hasattr(att, 'zaya_block'))
print('moe has self_attn:', hasattr(moe, 'self_attn'))
print('moe has zaya_block:', hasattr(moe, 'zaya_block'))
print('att.input_norm.weight:', att.input_norm.weight.shape)
print('att.res_scale.not_first_layer:', att.res_scale.not_first_layer)
"
```
Expected:
```
att has self_attn: True
att has zaya_block: False
moe has self_attn: False
moe has zaya_block: True
att.input_norm.weight: (2048,)
att.res_scale.not_first_layer: False
```

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 1 task 9: ZayaDecoderATTLayer + ZayaDecoderMLPLayer"
```

---

### Task 10: ZayaModel + Model wrapper + sanitize + registration

The final assembly: outer `Model` that mlx_lm.load instantiates.

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/__init__.py` (or wherever the model registry lives — see Step 4)

- [ ] **Step 1: Append `ZayaModel` and `Model`**

Append to `zaya.py`:
```python
class ZayaModel(nn.Module):
    """Embedding + 80 alternating decoder layers + final ResidualScaling + final RMSNorm.

    Per modular_zaya.py:1642-1956. Skeleton only.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = []
        for layer_n in range(args.num_hidden_layers):
            if layer_n % 2 == 1:
                self.layers.append(ZayaDecoderMLPLayer(args, layer_n))
            else:
                self.layers.append(ZayaDecoderATTLayer(args, layer_n))
        if args.scale_residual_merge:
            # Final residual scaling, layer_n = num_hidden_layers (always non-first)
            self.res_scale = ResidualScaling(args, args.num_hidden_layers)
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)


class Model(nn.Module):
    """The mlx-lm canonical wrapper. Forward is a stub for Phase 1."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ZayaModel(args)
        # tie_word_embeddings: lm_head is None; Phase 9 will use embed_tokens.as_linear.
        self.lm_head = None if args.tie_word_embeddings else nn.Linear(
            args.hidden_size, args.vocab_size, bias=args.lm_head_bias
        )

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self) -> int:
        return self.args.head_dim

    @property
    def n_kv_heads(self) -> int:
        return self.args.num_key_value_heads

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        # Phase 1 stub. Phase 9 will implement the real forward.
        raise NotImplementedError("Zaya forward is implemented in Phase 9; this skeleton supports weight loading only.")

    def sanitize(self, weights: dict) -> dict:
        """Remap HF safetensors keys to MLX keys.

        Steps:
          1. PyTorch Conv1d weights have shape (out, in/groups, kernel).
             MLX Conv1d weights have shape (out, kernel, in/groups).
             Transpose conv_qk.{0,1}.weight from (out, in/g, k) to (out, k, in/g).
          2. tie_word_embeddings means lm_head.weight is not in HF; nothing to drop.
          3. Pass through everything else.

        Logs any unexpected keys.
        """
        out = {}
        for k, v in weights.items():
            if "self_attn.qkv.conv_qk" in k and k.endswith(".weight"):
                # PyTorch Conv1d (out, in/g, kernel) -> MLX (out, kernel, in/g)
                out[k] = v.transpose(0, 2, 1)
            else:
                out[k] = v
        return out
```

- [ ] **Step 2: Smoke-test instantiation**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -c "
from mlx_lm.models.zaya import ModelArgs, Model
a = ModelArgs()
m = Model(a)
print('model_type:', m.model_type)
print('layer count:', len(m.layers))
print('layer 0 type:', type(m.layers[0]).__name__)
print('layer 1 type:', type(m.layers[1]).__name__)
print('lm_head:', m.lm_head)
print('embed_tokens.weight.shape:', m.model.embed_tokens.weight.shape)
print('final_norm.weight.shape:', m.model.final_norm.weight.shape)
print('model.res_scale.not_first_layer:', m.model.res_scale.not_first_layer)
"
```
Expected:
```
model_type: zaya
layer count: 80
layer 0 type: ZayaDecoderATTLayer
layer 1 type: ZayaDecoderMLPLayer
lm_head: None
embed_tokens.weight.shape: (262272, 2048)
final_norm.weight.shape: (2048,)
model.res_scale.not_first_layer: True
```

- [ ] **Step 3: Find the mlx-lm model registry**

Run:
```bash
cd ~/code/personal/mlx-lm
grep -rn "model_type" mlx_lm/utils.py mlx_lm/models/__init__.py 2>/dev/null | head -10
grep -rn "from . import" mlx_lm/models/__init__.py | head -5
```
The exact registration mechanism varies by mlx-lm version. Modern mlx-lm typically does dynamic import based on model_type, looking for `mlx_lm/models/<model_type>.py`. **No explicit registry edit needed** if zaya.py is in the right place — mlx_lm.load will find it by introspection.

If a registry pattern is found (older mlx-lm versions), add zaya there. Otherwise, no edit is needed.

- [ ] **Step 4: Run the integration test**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_weight_loading.py -v
```
Expected behaviors:
- `test_model_type_is_zaya`: PASS
- `test_layer_count`: PASS
- `test_layers_alternate_att_moe`: PASS
- `test_layer_0_att_shapes`: PASS
- `test_layer_1_moe_shapes_and_no_eda`: PASS
- `test_layer_3_moe_has_eda`: PASS
- `test_embed_and_lm_head_share_weights`: PASS
- `test_total_param_count`: may FAIL if any weight key is not loaded — see Task 11.

If `mlx_lm.load` raises an error about missing weights or unexpected keys, those errors are diagnostic — they tell us exactly which keys aren't being mapped. Note them down for Task 11.

- [ ] **Step 5: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py mlx_lm/models/__init__.py 2>/dev/null
git commit -m "Phase 1 task 10: ZayaModel + Model + sanitize + register"
```

---

### Task 11: Iterate `sanitize` until all weights load

The first run of the integration test almost certainly surfaces a few mismatches. Fix them in `sanitize` until the test is green.

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py` (`sanitize` method)

- [ ] **Step 1: Capture the precise error from Task 10 step 4**

Re-run the test with `--tb=long`:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_weight_loading.py -v --tb=long 2>&1 | tee /tmp/phase1-test.log | tail -40
```

- [ ] **Step 2: Categorize errors**

Common failure modes and their fixes:
- **"unexpected key foo.bar"**: weight key in safetensors that the skeleton doesn't have. Either drop in sanitize or add the parameter to the skeleton.
- **"missing key foo.bar"**: skeleton has a parameter that no safetensors weight maps to. Either drop the parameter (skeleton has spurious member) or add a sanitize rule that creates the key from another source (e.g., aliasing).
- **"shape mismatch"**: parameter shape in skeleton doesn't match the .safetensors tensor. Fix the skeleton.

For each mismatch, decide: skeleton bug (fix the class) vs sanitize gap (add a rule). When in doubt, fix the skeleton — sanitize should stay small.

- [ ] **Step 3: Apply fixes one at a time, re-running the test after each**

Iteration loop:
```bash
# After each fix:
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_weight_loading.py -v --tb=short 2>&1 | tail -20
```

Stop when all 8 tests pass.

- [ ] **Step 4: Final full run**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_weight_loading.py -v
```
Expected: 8/8 pass.

- [ ] **Step 5: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 1 task 11: iterate sanitize until 8/8 weight-loading tests pass"
```

---

### Task 12: Push fork + update STATUS

**Files:**
- Modify: `~/code/personal/zaya1-mlx/STATUS.md`

- [ ] **Step 1: Push the mlx-lm fork**

Run:
```bash
cd ~/code/personal/mlx-lm
git push origin zaya1
```
Expected: branch pushed to `https://github.com/zappleg8/mlx-lm`.

- [ ] **Step 2: Update STATUS.md**

Replace the "Current phase" + "What's done" + "What's next" sections of `~/code/personal/zaya1-mlx/STATUS.md` to reflect Phase 1 completion. The relevant updates:

- "Current phase" → `Phase 2 — partial RoPE wrapper (not yet started).`
- Add bullet under "What's done": `Phase 1 complete: ZAYA1 skeleton + weight loading in mlx-lm fork (~/code/personal/mlx-lm, branch zaya1). All 2,483 HF safetensors load with strict=True; param count matches HF total (~8.84B); tie_word_embeddings honored; 8/8 weight-loading tests pass.`
- "What's next" → describe Phase 2 (partial RoPE) and that a new plan needs to be written.

- [ ] **Step 3: Commit + push zaya1-mlx**

Run:
```bash
cd ~/code/personal/zaya1-mlx
git add STATUS.md
git commit -m "Phase 1 complete: status update"
git push origin main
```

---

## Phase 1 Gate Verification

Before declaring Phase 1 done, the following must be true:

- [ ] `cd ~/code/personal/zaya1-mlx/validation && .venv/bin/python -m pytest test_weight_loading.py -v` shows 8/8 passing.
- [ ] `cd ~/code/personal/mlx-lm && git log origin/zaya1 --oneline | head -5` shows commits pushed to GitHub.
- [ ] `~/code/personal/zaya1-mlx/reference/notes/hf-weight-keys.md` exists with the 28 unique key patterns.
- [ ] `~/code/personal/zaya1-mlx/reference/notes/mlx-lm-conventions.md` exists with conventions documented.
- [ ] `STATUS.md` reflects Phase 1 complete.
- [ ] No new errors or warnings on the existing reference dump tests.

If any fail, do not start writing Plan 3 (Phase 2) yet — fix the failure, then revalidate.

# Phase 4: ZayaAttention Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ZayaAttention.__call__` — composes CCA output (Q, K, V), reshapes into multi-head layout, applies partial RoPE, runs standard scaled dot product attention with GQA, and projects through `o_proj`. Verify per-tensor parity against PyTorch reference dumps for `self_attn_out` at layers 0, 40, 78.

**Architecture:** Most pieces are already validated primitives (CCA from Phase 3, partial RoPE from Phase 2, mlx-lm's `scaled_dot_product_attention`). This phase just composes them. The non-trivial bits: (a) reshape Q to 8 heads (the "hardcoded query compression" — half of `num_attention_heads`), (b) repeat KV by `num_key_value_groups // 2 = 4` to match 8 Q heads, (c) softmax in fp32 to match PyTorch eager attention. Cache integration (`use_cache=True` path) is included for Phase 7 readiness but tested only with `cache=None`.

**Tech Stack:** MLX 0.31.2, mlx-lm fork. No new dependencies.

---

## Pre-flight Setup

- [ ] **Confirm Phase 3 state**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation && \
  .venv/bin/python -m pytest -q 2>&1 | tail -2
```
Expected: 14 passed.

- [ ] **Confirm self_attn_out reference tensors present**

Run:
```bash
ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L0_self_attn_out.npy && \
  ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L40_self_attn_out.npy && \
  ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L78_self_attn_out.npy
```
Expected: all three exist.

---

### Task 1: Failing ZayaAttention forward parity test

**Files:**
- Create: `~/code/personal/zaya1-mlx/validation/test_attention_forward.py`

- [ ] **Step 1: Write the test**

Write `validation/test_attention_forward.py`:

```python
"""Phase 4 gate test: ZayaAttention forward parity against reference dumps.

Feeds each layer's input_norm_out directly into model.layers[i].self_attn
(skipping the residual stream) and compares the attention output to the
captured `self_attn_out` reference. cache=None throughout; cache integration
is exercised in Phase 7 generation tests.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"


# bf16 noise budget for full attention. Composes CCA noise + softmax + GQA
# matmuls + o_proj. Empirically settles around 0.05-0.2 absolute on the
# o_proj output (range typically -2 to 2).
ATT_OUT_TOL = 5e-1


@pytest.fixture(scope="session")
def loaded_model():
    from mlx_lm import load

    model, _tokenizer = load("Zyphra/ZAYA1-8B")
    return model


def _load_npy(name: str) -> mx.array:
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"Reference tensor {name} missing")
    return mx.array(np.load(path))


def _run_attn(loaded_model, layer_idx: int) -> mx.array:
    """Feed L{i}_input_norm_out into layer i's self_attn. Returns attn output."""
    if layer_idx % 2 != 0:
        raise ValueError(f"Layer {layer_idx} is a MoE layer, not ATT")
    hs = _load_npy(f"L{layer_idx}_input_norm_out.npy")  # (B, S, H)
    weight_dtype = loaded_model.layers[layer_idx].self_attn.qkv.linear_q.weight.dtype
    hs = hs.astype(weight_dtype)
    out = loaded_model.layers[layer_idx].self_attn(hs, mask="causal", cache=None)
    return out


def _compare(name: str, mlx_out: mx.array, ref_path: str, tol: float):
    ref = _load_npy(ref_path)
    if mlx_out.shape != ref.shape:
        pytest.fail(f"{name} shape mismatch: mlx={mlx_out.shape}, ref={ref.shape}")
    diff = float(mx.max(mx.abs(mlx_out.astype(mx.float32) - ref)))
    assert diff < tol, f"{name} max abs diff: {diff} (tol {tol})"


def test_attn_layer_0(loaded_model):
    out = _run_attn(loaded_model, 0)
    _compare("L0 self_attn_out", out, "L0_self_attn_out.npy", ATT_OUT_TOL)


def test_attn_layer_40(loaded_model):
    out = _run_attn(loaded_model, 40)
    _compare("L40 self_attn_out", out, "L40_self_attn_out.npy", ATT_OUT_TOL)


def test_attn_layer_78(loaded_model):
    out = _run_attn(loaded_model, 78)
    _compare("L78 self_attn_out", out, "L78_self_attn_out.npy", ATT_OUT_TOL)
```

- [ ] **Step 2: Confirm tests fail (no __call__ on ZayaAttention)**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_attention_forward.py -v --tb=short 2>&1 | tail -15
```
Expected: tests fail because `ZayaAttention` is not callable.

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add validation/test_attention_forward.py
git commit -m "Phase 4 task 1: failing ZayaAttention forward parity tests"
```

---

### Task 2: Implement ZayaAttention.__call__

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py` (add `__call__` to `ZayaAttention`)

- [ ] **Step 1: Add the imports needed**

Verify these are at the top of `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`. If `scaled_dot_product_attention` is not yet imported from `.base`, add it:

```python
from .base import BaseModelArgs, scaled_dot_product_attention
```

(If only `BaseModelArgs` is imported, append `, scaled_dot_product_attention` to the existing import line.)

- [ ] **Step 2: Add `__call__` to ZayaAttention**

Add this `__call__` method inside the `ZayaAttention` class, after `__init__`:

```python
    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        cca_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """ZayaAttention forward.

        Args:
          hidden_states: (B, S, H) — output of the layer's input_norm.
          mask: causal mask. Pass "causal" string for the default; pass an
            additive mask tensor of shape (B, 1, S, S) for custom shapes.
          cache: KV cache (mlx-lm style). For Phase 4 parity tests we use
            cache=None.
          cca_mask: optional padding mask threaded through to CCA.

        Returns: (B, S, hidden_size) attention output post o_proj.

        Implements modular_zaya.py:566-656 (eager path) in MLX.
        """
        B, S, _ = hidden_states.shape
        # CCA produces compressed Q (8 heads of 128), K and V (each 2 heads of 128).
        num_q_heads = self.qkv.num_q_heads  # 8
        num_kv_heads = self.qkv.num_kv_heads  # 2
        head_dim = self.qkv.head_dim  # 128
        # GQA group size: each KV head is repeated this many times to match Q.
        group_size = num_q_heads // num_kv_heads  # 4

        q_flat, k_flat, v_flat = self.qkv(
            hidden_states, cca_mask=cca_mask, past_key_values=None
        )

        # Reshape into (B, n_heads, S, D) for attention
        queries = q_flat.reshape(B, S, num_q_heads, head_dim).transpose(0, 2, 1, 3)
        keys = k_flat.reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        values = v_flat.reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # Apply partial RoPE. RoPE handles offset for KV-cache continuation.
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # GQA: repeat each KV head group_size times to match num_q_heads.
        # MLX's scaled_dot_product_attention handles GQA when n_q > n_kv;
        # we don't need to manually repeat. Pass them as-is.
        scale = head_dim ** -0.5
        attn_out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

        # (B, n_heads=8, S, D=128) → (B, S, n_heads * D = 1024)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, num_q_heads * head_dim)
        return self.o_proj(attn_out)
```

- [ ] **Step 3: Run the L0 test**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_attention_forward.py::test_attn_layer_0 -v --tb=short 2>&1 | tail -10
```
Expected: depending on bf16 noise propagation, may pass or fail with a specific diff value.

- [ ] **Step 4: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 4 task 2: ZayaAttention.__call__ forward implementation"
```

---

### Task 3: Iterate until L0 passes

If L0 fails, the most likely causes:

1. **Softmax precision**: PyTorch upcasts to fp32 explicitly. MLX's `scaled_dot_product_attention` may stay in bf16. If diff is large and concentrated where attention is sharp, this is the cause.
2. **Causal mask shape**: `"causal"` string needs MLX's helper. If that produces a different mask than PyTorch's `_update_causal_mask`, the values diverge. For S=7 with no padding, both should produce the same lower-triangular mask.
3. **GQA handling**: MLX's `scaled_dot_product_attention` is supposed to handle GQA automatically when `n_q_heads > n_kv_heads`. Verify by checking whether it's repeating KV correctly.
4. **RoPE offset**: when cache=None we pass no offset; this should be equivalent to offset=0.

- [ ] **Step 1: If L0 fails, run a diagnostic comparison**

```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python <<'PY'
import mlx.core as mx
import numpy as np
from pathlib import Path
from mlx_lm import load

model, _ = load("Zyphra/ZAYA1-8B")
ref_dir = Path("../reference/activations/smoke")
hs = mx.array(np.load(ref_dir / "L0_input_norm_out.npy")).astype(
    model.layers[0].self_attn.qkv.linear_q.weight.dtype
)
out = model.layers[0].self_attn(hs, mask="causal", cache=None)
ref = mx.array(np.load(ref_dir / "L0_self_attn_out.npy"))
print("attn_out shape:", out.shape, "ref shape:", ref.shape)
print("max abs diff:", float(mx.max(mx.abs(out.astype(mx.float32) - ref))))
print("ref range:", float(mx.min(ref)), float(mx.max(ref)))

# Also check intermediate o_proj input
qkv_q, qkv_k, qkv_v = model.layers[0].self_attn.qkv(hs, cca_mask=None, past_key_values=None)
print("qkv outputs match Phase 3 expectations: q.shape =", qkv_q.shape)
PY
```

- [ ] **Step 2: Apply targeted fixes**

Common fixes (apply only the relevant one):

**Fix A — fp32 softmax**: replace the `scaled_dot_product_attention` call with manual attention to control softmax dtype:
```python
scale = head_dim ** -0.5
# Manual GQA: tile keys/values to match num_q_heads
keys_g = mx.repeat(keys, group_size, axis=1)  # (B, n_q, S, D)
values_g = mx.repeat(values, group_size, axis=1)
attn_scores = (queries * scale) @ keys_g.transpose(0, 1, 3, 2)  # (B, n_q, S_q, S_k)
if mask == "causal":
    qL, kL = attn_scores.shape[-2:]
    causal = mx.arange(kL - qL, kL)[:, None] >= mx.arange(kL)[None]
    attn_scores = mx.where(causal, attn_scores, mx.finfo(attn_scores.dtype).min)
elif isinstance(mask, mx.array):
    attn_scores = attn_scores + mask
attn_probs = mx.softmax(attn_scores.astype(mx.float32), axis=-1).astype(attn_scores.dtype)
attn_out = attn_probs @ values_g  # (B, n_q, S, D)
```

**Fix B — Mask shape**: if MLX's `"causal"` mask doesn't broadcast to the actual shape, build it explicitly:
```python
mask_arr = mx.full((S, S), -mx.inf)
mask_arr = mx.triu(mask_arr, k=1)
mask_arr = mask_arr[None, None]  # (1, 1, S, S)
```
Pass `mask_arr` instead of `"causal"`.

- [ ] **Step 3: Verify L0 passes**

Run:
```bash
.venv/bin/python -m pytest test_attention_forward.py::test_attn_layer_0 -v
```
Expected: PASS.

- [ ] **Step 4: Commit any fixes**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 4 task 3: ZayaAttention forward — iterate to L0 parity"
```
(Skip if no fix was needed.)

---

### Task 4: Verify L40 and L78 also pass

- [ ] **Step 1: Run all three layer tests**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_attention_forward.py -v
```
Expected: 3/3 pass.

- [ ] **Step 2: Run the full validation suite**

Run:
```bash
.venv/bin/python -m pytest -q
```
Expected: 17 passed (8 weight loading + 3 partial RoPE + 3 CCA forward + 3 attention forward).

---

### Task 5: Push fork + update STATUS

**Files:**
- Modify: `~/code/personal/zaya1-mlx/STATUS.md`

- [ ] **Step 1: Push the mlx-lm fork**

```bash
cd ~/code/personal/mlx-lm
git push origin zaya1
```

- [ ] **Step 2: Update STATUS.md**

Replace the "Current phase" section and append to "What's done":

- "Current phase" → `Phase 5 — ResidualScaling + ZayaDecoderATTLayer forward (composes residual stream + input_norm + attention).`
- Append to "What's done":
  ```
  Phase 4 (ZayaAttention forward):
    - ZayaAttention.__call__ implemented: CCA → reshape to multi-head →
      partial RoPE → standard scaled dot product attention (mlx-lm's helper) →
      reshape → o_proj
    - 3/3 attention forward parity tests pass at L0, L40, L78
    - Validation suite at 17/17
  ```
- "What's next" → describe Phase 5 (full ATT decoder layer) and that a new plan needs to be written.

- [ ] **Step 3: Commit and push**

```bash
cd ~/code/personal/zaya1-mlx
git add STATUS.md
git commit -m "Phase 4 complete: status update"
git push origin main
```

---

## Phase 4 Gate Verification

- [ ] `cd ~/code/personal/zaya1-mlx/validation && .venv/bin/python -m pytest -q` shows 17 passed.
- [ ] `cd ~/code/personal/mlx-lm && git log origin/zaya1 --oneline | head -3` shows the Phase 4 commit pushed.
- [ ] `STATUS.md` reflects Phase 4 complete.
- [ ] Phase 0 dump tests still pass: `cd ~/code/personal/zaya1-mlx/reference && .venv/bin/python -m pytest -q` shows 5 passed.

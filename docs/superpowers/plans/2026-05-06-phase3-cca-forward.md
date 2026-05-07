# Phase 3: CCA Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `CCA.__call__` (the Compressed Causal Attention forward pass) and verify per-tensor parity against PyTorch reference dumps for `self_attn_qkv_q`, `_k`, `_v` at layers 0, 40, and 78. This is the most architecturally novel piece of the model — depthwise 1D causal conv on Q+K + mean residual mixing + two-stream V + per-head L2 norm with temperature.

**Architecture:** All math implemented in MLX in HF layout `(B, S, H)` throughout (PyTorch's reference uses `[S, B, H]` internally with mid-function transposes; we keep one consistent layout). Two-stage `nn.Conv1d` chain handles the depthwise+grouped causal conv. The time-shifted V₂ stream comes from `pad(hs[:, :-1], front=1)` in prefill mode. We feed each layer's `input_norm_out` (captured by Phase 0 dumps) directly into `model.layers[i].self_attn.qkv`, bypassing the residual stream — that's tested independently in Phase 5. Tolerances follow Phase 2's `BF16_*` constants since reference tensors are bf16-rounded.

**Tech Stack:** MLX 0.31.2, mlx-lm fork. No new dependencies.

---

## Pre-flight Setup

- [ ] **Confirm Phase 2 state**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation && \
  .venv/bin/python -m pytest -q 2>&1 | tail -2
```
Expected: 11 passed.

- [ ] **Confirm CCA reference tensors present**

Run:
```bash
ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L0_self_attn_qkv_{q,k,v}.npy && \
  ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L0_input_norm_out.npy
```
Expected: 4 files exist.

---

### Task 1: Failing CCA forward parity test (L0 only)

**Files:**
- Create: `~/code/personal/zaya1-mlx/validation/test_cca_forward.py`

- [ ] **Step 1: Write the test**

Write `validation/test_cca_forward.py`:

```python
"""Phase 3 gate test: CCA forward parity against reference dumps.

Feeds each layer's input_norm_out directly into model.layers[i].self_attn.qkv
and compares the (Q, K, V) output to PyTorch's captured tensors. Bypasses the
residual stream entirely (tested separately in Phase 5).
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"


# Tolerances follow Phase 2's bf16-aware constants.
# CCA's L2 norm + temperature can amplify bf16 noise modestly.
CCA_TOL_Q = 5e-2
CCA_TOL_K = 5e-2
CCA_TOL_V = 1e-2


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


def _run_cca(loaded_model, layer_idx: int):
    """Feed L{i}_input_norm_out into layer i's CCA. Returns (q, k, v) MLX arrays."""
    if layer_idx % 2 != 0:
        raise ValueError(f"Layer {layer_idx} is a MoE layer, not ATT")

    hs = _load_npy(f"L{layer_idx}_input_norm_out")  # (B, S, H)
    # Cast to the model dtype to match what the layer would receive in a
    # real forward pass.
    weight_dtype = loaded_model.layers[layer_idx].self_attn.qkv.linear_q.weight.dtype
    hs = hs.astype(weight_dtype)

    cca = loaded_model.layers[layer_idx].self_attn.qkv
    q, k, v = cca(hs, cca_mask=None, past_key_values=None)
    return q, k, v


def _compare(name: str, mlx_out: mx.array, ref_path: str, tol: float):
    ref = _load_npy(ref_path)
    if mlx_out.shape != ref.shape:
        pytest.fail(f"{name} shape mismatch: mlx={mlx_out.shape}, ref={ref.shape}")
    diff = float(mx.max(mx.abs(mlx_out.astype(mx.float32) - ref)))
    assert diff < tol, f"{name} max abs diff: {diff} (tol {tol})"


def test_cca_layer_0(loaded_model):
    q, k, v = _run_cca(loaded_model, 0)
    _compare("L0 Q", q, "L0_self_attn_qkv_q.npy", CCA_TOL_Q)
    _compare("L0 K", k, "L0_self_attn_qkv_k.npy", CCA_TOL_K)
    _compare("L0 V", v, "L0_self_attn_qkv_v.npy", CCA_TOL_V)


def test_cca_layer_40(loaded_model):
    q, k, v = _run_cca(loaded_model, 40)
    _compare("L40 Q", q, "L40_self_attn_qkv_q.npy", CCA_TOL_Q)
    _compare("L40 K", k, "L40_self_attn_qkv_k.npy", CCA_TOL_K)
    _compare("L40 V", v, "L40_self_attn_qkv_v.npy", CCA_TOL_V)


def test_cca_layer_78(loaded_model):
    q, k, v = _run_cca(loaded_model, 78)
    _compare("L78 Q", q, "L78_self_attn_qkv_q.npy", CCA_TOL_Q)
    _compare("L78 K", k, "L78_self_attn_qkv_k.npy", CCA_TOL_K)
    _compare("L78 V", v, "L78_self_attn_qkv_v.npy", CCA_TOL_V)
```

- [ ] **Step 2: Confirm tests fail because CCA.__call__ is not implemented**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_cca_forward.py -v --tb=short 2>&1 | tail -15
```
Expected: tests fail because `CCA` has no `__call__` (or raises NotImplementedError). The error message will identify the missing method.

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add validation/test_cca_forward.py
git commit -m "Phase 3 task 1: failing CCA forward parity tests (L0, L40, L78)"
```

---

### Task 2: Implement CCA.__call__

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py` (add `__call__` to `CCA` class)

- [ ] **Step 1: Add `__call__` to CCA**

Add this `__call__` method inside the `CCA` class (after `__init__`):

```python
    def __call__(
        self,
        hidden_states: mx.array,
        cca_mask: mx.array | None = None,
        past_key_values=None,
    ):
        """Compressed Causal Attention forward.

        Args:
          hidden_states: (B, S, H) input from the layer's input_norm.
          cca_mask: optional (B, S) attention mask; multiplied into hidden_states
            during prefill. If None, treated as all-ones (no masking).
          past_key_values: ZayaDynamicCache for generation. If None, prefill
            mode only — no conv state cache, no prev_hs cache.

        Returns: (Q, K, V) flattened to HF layout (B, S, n_heads * head_dim).
          Q: (B, S, num_q_heads * head_dim) = (B, S, 1024)
          K: (B, S, num_kv_heads * head_dim) = (B, S, 256)
          V: (B, S, num_kv_heads * head_dim) = (B, S, 256)

        Implements the math of modular_zaya.py:370-521 in HF (B, S, H) layout.
        Generation-with-cache (`has_previous_state`) is deferred to Phase 4.
        """
        B, S, _ = hidden_states.shape
        gqa_groups = self.num_q_heads // self.num_kv_heads  # 4
        sqrt_head_dim = mx.array(self.head_dim ** 0.5, dtype=hidden_states.dtype)

        # Apply cca_mask during prefill if S > 1 and mask is provided.
        if cca_mask is not None and S > 1:
            hidden_states = hidden_states * cca_mask[:, :, None]

        # ---- Linear projections ----
        q = self.linear_q(hidden_states)  # (B, S, 1024)
        k = self.linear_k(hidden_states)  # (B, S, 256)
        qk_packed0 = mx.concatenate([q, k], axis=-1)  # (B, S, 1280)

        # ---- Pre-conv mean residual ----
        query_pre = q.reshape(B, S, self.num_q_heads, self.head_dim)
        key_pre_kv = k.reshape(B, S, self.num_kv_heads, self.head_dim)
        # Repeat each KV head gqa_groups=4 times to align with num_q_heads=8.
        key_pre = mx.expand_dims(key_pre_kv, axis=-2)  # (B, S, 2, 1, 128)
        key_pre = mx.repeat(key_pre, gqa_groups, axis=-2)  # (B, S, 2, 4, 128)
        key_pre = key_pre.reshape(B, S, self.num_q_heads, self.head_dim)
        qk_mean_q = (query_pre + key_pre) / 2  # (B, S, 8, 128)
        qk_mean_k = qk_mean_q.reshape(
            B, S, self.num_kv_heads, gqa_groups, self.head_dim
        ).mean(axis=-2)  # (B, S, 2, 128)

        # ---- Two-stage causal conv ----
        # MLX nn.Conv1d expects (B, S, C); pad sequence axis on front.
        total_padding = (self.cca_time0 - 1) + (self.cca_time1 - 1)  # 2
        qk_padded = mx.pad(qk_packed0, [(0, 0), (total_padding, 0), (0, 0)])
        qk_packed3 = self.conv_qk(qk_padded)  # (B, S, 1280)

        # ---- Build queries/keys from conv output + means ----
        query = qk_packed3[..., : self.latent_q_dim].reshape(
            B, S, self.num_q_heads, self.head_dim
        ) + qk_mean_q  # (B, S, 8, 128)
        key = qk_packed3[..., self.latent_q_dim :].reshape(
            B, S, self.num_kv_heads, self.head_dim
        ) + qk_mean_k  # (B, S, 2, 128)

        # ---- Two-stream V ----
        v1 = self.val_proj1(hidden_states)  # (B, S, 128)
        # Time-shifted hidden states: drop last token, prepend a zero at front.
        # Equivalent to PyTorch's F.pad(hs[:-1], (0,0, 0,0, 1,0)) in [S,B,H] layout.
        if S > 1:
            hs_shifted = mx.pad(hidden_states[:, :-1], [(0, 0), (1, 0), (0, 0)])
        else:
            hs_shifted = mx.zeros_like(hidden_states)
        v2 = self.val_proj2(hs_shifted)  # (B, S, 128)
        value = mx.concatenate([v1, v2], axis=-1).reshape(
            B, S, self.num_kv_heads, self.head_dim
        )  # (B, S, 2, 128)

        # ---- L2 normalize Q and K, apply per-head temperature ----
        # query: (B, S, 8, 128); query_norm: (B, S, 8, 1)
        query_norm = mx.linalg.norm(
            query.astype(mx.float32), axis=-1, keepdims=True
        ).astype(query.dtype)
        key_norm = mx.linalg.norm(
            key.astype(mx.float32), axis=-1, keepdims=True
        ).astype(key.dtype)
        query = query * (sqrt_head_dim / query_norm)
        # temp shape (num_kv_heads,) → (1, 1, num_kv_heads, 1) for broadcast
        temp_b = self.temp[None, None, :, None]
        key = key * (sqrt_head_dim / key_norm) * temp_b

        # ---- Flatten head axis to HF flat layout ----
        query = query.reshape(B, S, self.num_q_heads * self.head_dim)
        key = key.reshape(B, S, self.num_kv_heads * self.head_dim)
        value = value.reshape(B, S, self.num_kv_heads * self.head_dim)

        return query, key, value
```

- [ ] **Step 2: Run the L0 test only first (fastest feedback)**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_cca_forward.py::test_cca_layer_0 -v --tb=short 2>&1 | tail -15
```
Expected: depending on the bf16 noise, may pass or fail. If it fails, the diff value points at the specific issue:
- Diff in Q only (not V): conv path or mean residual issue
- Diff in V only: val_proj/time-shift issue
- All three small but exceeding tolerance: dtype/precision; consider fp32 in linalg.norm
- All three large: layout/transpose issue

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 3 task 2: CCA.__call__ forward implementation"
```

---

### Task 3: Iterate until L0 passes

The reference dump has cca_time0=2, cca_time1=2 hardcoded into the conv kernels. If the L0 test fails, the most likely causes are:

1. **Conv weight transpose**: `sanitize` already transposes conv_qk weights from PyTorch (out, in/g, k) to MLX (out, k, in/g). If diff is large, double-check this — print the transposed weight shape and verify it matches what nn.Conv1d expects.
2. **Sequence padding direction**: PyTorch's `F.pad((total_padding, 0))` on the LAST dim of `[B, E, S]` pads the sequence FRONT. In MLX HF layout `[B, S, E]`, the equivalent is padding axis=1 with `(total_padding, 0)`. Verify with a small synthetic test.
3. **L2 norm precision**: bf16 norm computation has precision issues. The implementation already upcasts to fp32 for the norm. If still failing, also do the multiplication in fp32 and downcast at the end.
4. **Per-head temperature broadcasting**: `temp` shape (2,) broadcasts as `[None, None, :, None]` → (1, 1, 2, 1). Verify this matches K shape (B, S, 2, 128).

- [ ] **Step 1: Run with extra diagnostics**

If L0 fails, run a diagnostic script:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python <<'PY'
import mlx.core as mx
import numpy as np
from pathlib import Path
from mlx_lm import load

model, _ = load("Zyphra/ZAYA1-8B")
ref_dir = Path("../reference/activations/smoke")

# Load input
hs = mx.array(np.load(ref_dir / "L0_input_norm_out.npy"))
weight_dtype = model.layers[0].self_attn.qkv.linear_q.weight.dtype
hs = hs.astype(weight_dtype)

cca = model.layers[0].self_attn.qkv
q, k, v = cca(hs, cca_mask=None, past_key_values=None)

# Compare each at fp32
q_ref = mx.array(np.load(ref_dir / "L0_self_attn_qkv_q.npy"))
k_ref = mx.array(np.load(ref_dir / "L0_self_attn_qkv_k.npy"))
v_ref = mx.array(np.load(ref_dir / "L0_self_attn_qkv_v.npy"))
print("Q max abs diff:", float(mx.max(mx.abs(q.astype(mx.float32) - q_ref))))
print("K max abs diff:", float(mx.max(mx.abs(k.astype(mx.float32) - k_ref))))
print("V max abs diff:", float(mx.max(mx.abs(v.astype(mx.float32) - v_ref))))

# Compare intermediates
linear_q_ref = mx.array(np.load(ref_dir / "L0_self_attn_qkv_linear_q_out.npy"))
# linear_q_out is in [S, B, H] in PyTorch; reshape to [B, S, H]
linear_q_ref_hf = linear_q_ref.transpose(1, 0, 2)
linear_q_mlx = cca.linear_q(hs)
print("linear_q diff:", float(mx.max(mx.abs(linear_q_mlx.astype(mx.float32) - linear_q_ref_hf))))
PY
```

The diagnostic compares `linear_q` output as a sanity check that the projection alone is correct. If `linear_q` matches but Q output doesn't, the bug is later in the pipeline (conv, mean, or norm).

- [ ] **Step 2: Apply targeted fixes based on the diagnostic**

Common fixes (apply only the relevant one):

**Fix A — Conv input/output layout swap:** if the conv output diff is large but linear_q matches, the conv is using the wrong layout. Verify with:
```python
print("conv_qk[0] weight shape:", cca.conv_qk.layers[0].weight.shape)  # expect (1280, 2, 1)
```

**Fix B — Sequence padding wrong side:** invert padding to `[(0,0), (0, total_padding), (0,0)]` if the conv output is right-shifted instead of left-shifted (compare against `L0_self_attn_qkv_conv_qk_1_out.npy`).

**Fix C — fp32 throughout the L2 norm step:** if Q/K diffs are 2-3x the tolerance and V is fine, upcast everything in the L2 norm step:
```python
q_fp32 = query.astype(mx.float32)
q_norm = mx.linalg.norm(q_fp32, axis=-1, keepdims=True)
query = (q_fp32 * (sqrt_head_dim / q_norm)).astype(query.dtype)
```

- [ ] **Step 3: Verify L0 passes**

Run:
```bash
.venv/bin/python -m pytest test_cca_forward.py::test_cca_layer_0 -v
```
Expected: PASS.

- [ ] **Step 4: Commit any fixes**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 3 task 3: CCA.__call__ — iterate to L0 parity"
```

(Skip this commit if no fix was needed.)

---

### Task 4: Verify L40 and L78 also pass

If L0 passes, L40 and L78 should follow without code changes — they exercise the same forward path with different weights. We test them to catch any layer-index-dependent bugs (e.g., temp scaling for layers far from init).

- [ ] **Step 1: Run all three layer tests**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_cca_forward.py -v
```
Expected: 3/3 pass. If L40 or L78 fails but L0 passed, the diff likely indicates accumulated precision drift or a layer-specific weight pattern; investigate via the same diagnostic script with `layers[40]` / `layers[78]`.

- [ ] **Step 2: Run the full validation suite**

Run:
```bash
.venv/bin/python -m pytest -q
```
Expected: 14 passed (8 weight loading + 3 partial RoPE + 3 CCA forward).

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

Replace the "Current phase" + "What's done" + "What's next" sections of `~/code/personal/zaya1-mlx/STATUS.md`:

- "Current phase" → `Phase 4 — ZayaAttention forward (CCA wrap + standard SDPA + GQA + RoPE + KV cache).`
- Append to "What's done":
  ```
  Phase 3 (CCA forward):
    - CCA.__call__ implemented in HF (B, S, H) layout: linear_q/k projections,
      pre-conv mean residual, two-stage depthwise/grouped Conv1d, two-stream V,
      per-head L2-normalized Q/K with learnable temperature
    - 3/3 CCA forward parity tests pass at layers 0, 40, 78 within bf16 tolerance
    - Validation suite at 14/14 (8 weight loading + 3 partial RoPE + 3 CCA)
  ```
- "What's next" → describe Phase 4 and that a new plan needs to be written.

- [ ] **Step 3: Commit and push**

```bash
cd ~/code/personal/zaya1-mlx
git add STATUS.md
git commit -m "Phase 3 complete: status update"
git push origin main
```

---

## Phase 3 Gate Verification

- [ ] `cd ~/code/personal/zaya1-mlx/validation && .venv/bin/python -m pytest -q` shows 14 passed (Phase 1: 8, Phase 2: 3, Phase 3: 3).
- [ ] `cd ~/code/personal/mlx-lm && git log origin/zaya1 --oneline | head -3` shows the Phase 3 commit pushed.
- [ ] `STATUS.md` reflects Phase 3 complete.
- [ ] Phase 0 dump_activations tests still pass: `cd ~/code/personal/zaya1-mlx/reference && .venv/bin/python -m pytest -q` shows 5 passed.

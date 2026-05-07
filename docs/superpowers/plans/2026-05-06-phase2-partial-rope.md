# Phase 2: Partial RoPE Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that mlx-lm's `nn.RoPE(dims=64, base=5e6, traditional=False)` produces output numerically equivalent to PyTorch's `apply_rotary_pos_emb` for Zaya's specific config, then bake the RoPE primitive into the ZayaAttention skeleton.

**Architecture:** mlx-lm already supports partial RoPE natively — `nn.RoPE(dims=N)` rotates only the first N features and passes the rest through unchanged. Our Glm4-derived ZayaRotaryEmbedding maps directly. The only risk is that Zaya's specific parameter combination (dims=64, base=5e6, traditional=False, no scaling) interacts oddly somewhere. We validate via a parity test against the PyTorch `apply_rotary_pos_emb` math, run on a synthetic input plus on the captured CCA Q/K from the smoke reference dump.

**Tech Stack:** MLX 0.31.2, mlx-lm fork. Validation venv only — no new PyTorch dependency. The PyTorch math is reimplemented in MLX (a 5-line function) so validation stays in the validation venv.

---

## Pre-flight Setup

- [ ] **Confirm Phase 1 state**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation && \
  .venv/bin/python -m pytest test_weight_loading.py -q 2>&1 | tail -3
```
Expected: 8 passed.

- [ ] **Confirm reference dumps still present**

Run:
```bash
ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L0_self_attn_qkv_q.npy && \
  ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L0_self_attn_qkv_k.npy && \
  ls ~/code/personal/zaya1-mlx/reference/activations/smoke/global_model_rotary_emb_out.npy
```
Expected: all three exist.

---

### Task 1: Augment dump_activations to capture both cos and sin

The current `_save_output` heuristic saves only the first tensor of a 2-tuple, which loses `sin` from the rotary embedding. Fix this by saving 2-tuple outputs as `_0` / `_1` while preserving the 3-tuple `_q`/`_k`/`_v` special case.

**Files:**
- Modify: `~/code/personal/zaya1-mlx/reference/dump_activations.py`

- [ ] **Step 1: Update `_save_output` to handle 2-tuples**

Replace the `_save_output` function in `reference/dump_activations.py` with:

```python
def _save_output(captured: Dict[str, np.ndarray], key_base: str, output) -> None:
    """Save whatever a forward hook saw — handle tensor, tuple, or ModelOutput.

    Conventions:
      - Plain tensor: saved as `{key_base}_out`
      - 3-tuple of same-dtype tensors (CCA Q,K,V): saved as `_q`, `_k`, `_v`
      - 2-tuple of same-dtype tensors (e.g. rotary cos,sin): saved as `_0`, `_1`
      - Other tuples: save first tensor as `_out`
      - ModelOutput dataclass: unwrap last_hidden_state or logits
    """
    if isinstance(output, torch.Tensor):
        captured[f"{key_base}_out"] = _to_numpy(output)
    elif isinstance(output, (tuple, list)):
        tensors = [x for x in output if isinstance(x, torch.Tensor)]
        if len(tensors) == 3 and tensors[0].dtype == tensors[1].dtype == tensors[2].dtype:
            captured[f"{key_base}_q"] = _to_numpy(tensors[0])
            captured[f"{key_base}_k"] = _to_numpy(tensors[1])
            captured[f"{key_base}_v"] = _to_numpy(tensors[2])
        elif len(tensors) == 2 and tensors[0].dtype == tensors[1].dtype:
            captured[f"{key_base}_0"] = _to_numpy(tensors[0])
            captured[f"{key_base}_1"] = _to_numpy(tensors[1])
        elif len(tensors) > 0:
            captured[f"{key_base}_out"] = _to_numpy(tensors[0])
    elif hasattr(output, "last_hidden_state") and isinstance(
        output.last_hidden_state, torch.Tensor
    ):
        captured[f"{key_base}_out"] = _to_numpy(output.last_hidden_state)
    elif hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
        captured[f"{key_base}_logits"] = _to_numpy(output.logits)
```

- [ ] **Step 2: Re-run smoke dump to pick up cos+sin**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python dump_activations.py --prompt-id smoke 2>&1 | tail -3
```
Expected: `Captured 3047 tensors` (one more than before — the new `_1` from rotary, replacing the old single `_out`).

- [ ] **Step 3: Verify cos and sin both saved**

Run:
```bash
ls ~/code/personal/zaya1-mlx/reference/activations/smoke/global_model_rotary_emb_*
.venv/bin/python -c "
import numpy as np
cos = np.load('activations/smoke/global_model_rotary_emb_0.npy')
sin = np.load('activations/smoke/global_model_rotary_emb_1.npy')
print('cos shape:', cos.shape, 'mean:', float(cos.mean()))
print('sin shape:', sin.shape, 'mean:', float(sin.mean()))
print('cos[0,0,:4]:', cos[0,0,:4])
print('sin[0,0,:4]:', sin[0,0,:4])
"
```
Expected: both shape `(1, 7, 64)`, dtype float32. cos has mean near 1.0 at position 0; sin has mean near 0.0 at position 0.

- [ ] **Step 4: Re-run the dump_activations tests to confirm nothing regressed**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -m pytest test_dump_activations.py -q
```
Expected: 5 passed.

- [ ] **Step 5: Re-run reasoning_short and long_context_seed for completeness**

Run:
```bash
.venv/bin/python dump_activations.py --prompt-id reasoning_short 2>&1 | tail -2
.venv/bin/python dump_activations.py --prompt-id long_context_seed 2>&1 | tail -2
```
Expected: each prints `Captured 3047 tensors` and the output dir.

- [ ] **Step 6: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/dump_activations.py
git commit -m "Phase 2 task 1: dump 2-tuple outputs as _0/_1 (captures rotary sin)"
```

---

### Task 2: Failing partial-RoPE parity test

We compare MLX's `nn.RoPE` to a hand-coded MLX implementation of PyTorch's `apply_rotary_pos_emb`. Both should produce numerically equivalent output for Zaya's config.

**Files:**
- Create: `~/code/personal/zaya1-mlx/validation/test_partial_rope.py`

- [ ] **Step 1: Write the test**

Write `validation/test_partial_rope.py`:

```python
"""Phase 2 gate test: verify mlx-lm nn.RoPE matches PyTorch's apply_rotary_pos_emb
for Zaya's specific config (dims=64, base=5e6, traditional=False).

Two complementary checks:
  1. Synthetic-input parity: build Q with known values, apply both paths,
     verify max abs diff < 1e-5.
  2. Reference-anchored parity: load captured pre-RoPE Q/K from the smoke
     reference dump + the captured cos/sin, apply both paths, verify match.

The 'PyTorch math' is reimplemented in MLX (the function is small) so we
don't need to import torch in the validation venv.
"""
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"


# Zaya config-derived constants
HEAD_DIM = 128
PARTIAL_ROTARY_FACTOR = 0.5
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64
ROPE_THETA = 5_000_000.0


def rotate_half_mlx(x: mx.array) -> mx.array:
    """Match modular_zaya.py:178-182. NeoX-style half-rotation."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_mlx(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    """Match modular_zaya.py:185-210 exactly.

    cos/sin shapes: (B, S, rotary_dim). Will be unsqueezed to broadcast
    across heads (axis=1 in HF q/k layout (B, H, S, D)).
    """
    rotary_dim = cos.shape[-1]
    cos = mx.expand_dims(cos, axis=1)  # (B, 1, S, rotary_dim)
    sin = mx.expand_dims(sin, axis=1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half_mlx(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half_mlx(k_rot) * sin)
    return mx.concatenate([q_rot, q_pass], axis=-1), mx.concatenate([k_rot, k_pass], axis=-1)


def _compute_zaya_cos_sin(seq_len: int) -> tuple[mx.array, mx.array]:
    """Reproduce the cos/sin tensors that Zaya's RotaryEmbedding produces.

    Standard frequency table: theta_i = 1 / base ** (2i/dim) for i in 0..dim/2.
    cos/sin then evaluated at positions 0..seq_len-1, with each frequency
    duplicated to fill the full rotary_dim (NeoX layout).
    """
    inv_freq = 1.0 / (ROPE_THETA ** (mx.arange(0, ROTARY_DIM, 2, dtype=mx.float32) / ROTARY_DIM))
    pos = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.expand_dims(pos, 1) * mx.expand_dims(inv_freq, 0)  # (S, rotary_dim/2)
    emb = mx.concatenate([freqs, freqs], axis=-1)  # NeoX: duplicate the half
    cos = mx.cos(emb)[None]  # (1, S, rotary_dim)
    sin = mx.sin(emb)[None]
    return cos, sin


def test_synthetic_input_parity():
    """Hand-coded apply_rotary_pos_emb_mlx vs mlx-lm nn.RoPE on a synthetic Q."""
    rng = np.random.default_rng(0)
    B, H, S, D = 1, 8, 7, HEAD_DIM
    q_np = rng.standard_normal((B, H, S, D)).astype(np.float32)
    k_np = rng.standard_normal((B, 2, S, D)).astype(np.float32)
    q = mx.array(q_np)
    k = mx.array(k_np)

    cos, sin = _compute_zaya_cos_sin(S)
    q_ref, k_ref = apply_rotary_pos_emb_mlx(q, k, cos, sin)

    rope = nn.RoPE(dims=ROTARY_DIM, base=ROPE_THETA, traditional=False)
    q_mlx = rope(q)
    k_mlx = rope(k)

    q_max_diff = float(mx.max(mx.abs(q_ref - q_mlx)))
    k_max_diff = float(mx.max(mx.abs(k_ref - k_mlx)))
    assert q_max_diff < 1e-5, f"Q diff: {q_max_diff}"
    assert k_max_diff < 1e-5, f"K diff: {k_max_diff}"


def test_against_dumped_cos_sin():
    """Use the cos/sin captured from PyTorch and verify the math reproduces correctly."""
    cos_path = REFERENCE_DIR / "global_model_rotary_emb_0.npy"
    sin_path = REFERENCE_DIR / "global_model_rotary_emb_1.npy"
    if not cos_path.exists() or not sin_path.exists():
        pytest.skip("Run dump_activations with the Phase-2 _save_output fix first")

    cos = mx.array(np.load(cos_path))  # (1, S, 64)
    sin = mx.array(np.load(sin_path))
    S = cos.shape[1]

    cos_local, sin_local = _compute_zaya_cos_sin(S)
    cos_diff = float(mx.max(mx.abs(cos - cos_local)))
    sin_diff = float(mx.max(mx.abs(sin - sin_local)))
    assert cos_diff < 1e-5, f"cos diff vs locally computed: {cos_diff}"
    assert sin_diff < 1e-5, f"sin diff vs locally computed: {sin_diff}"


def test_against_dumped_q():
    """End-to-end: load captured pre-RoPE Q (CCA output), apply both RoPE paths,
    confirm equivalence."""
    q_path = REFERENCE_DIR / "L0_self_attn_qkv_q.npy"
    cos_path = REFERENCE_DIR / "global_model_rotary_emb_0.npy"
    sin_path = REFERENCE_DIR / "global_model_rotary_emb_1.npy"
    if not (q_path.exists() and cos_path.exists() and sin_path.exists()):
        pytest.skip("Reference dump artifacts missing")

    q_pre = np.load(q_path)  # (B=1, S=7, num_q_heads*head_dim=1024)
    cos = mx.array(np.load(cos_path))
    sin = mx.array(np.load(sin_path))
    B, S, _ = q_pre.shape
    num_q_heads = 8
    q = mx.array(q_pre).reshape(B, S, num_q_heads, HEAD_DIM).transpose(0, 2, 1, 3)
    k_dummy = mx.zeros((B, 2, S, HEAD_DIM))

    q_ref, _ = apply_rotary_pos_emb_mlx(q, k_dummy, cos, sin)

    rope = nn.RoPE(dims=ROTARY_DIM, base=ROPE_THETA, traditional=False)
    q_mlx = rope(q)

    diff = float(mx.max(mx.abs(q_ref - q_mlx)))
    assert diff < 1e-4, f"max abs diff: {diff}"
```

- [ ] **Step 2: Run the test, confirm it passes (no implementation needed yet)**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_partial_rope.py -v
```
Expected: All 3 tests pass. If `test_against_dumped_cos_sin` fails, the local cos/sin computation in the test doesn't match Zaya's exact convention — investigate via the diff value.

If a test fails, the most likely causes:
- Different `traditional` convention (try `traditional=True` in `nn.RoPE`).
- Different cos/sin frequency layout (NeoX vs interleaved).

**Stop and fix before proceeding to Task 3.**

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add validation/test_partial_rope.py
git commit -m "Phase 2 task 2: partial RoPE parity tests (synthetic + reference-anchored)"
```

---

### Task 3: Add RoPE primitive to ZayaAttention skeleton

The skeleton currently defines no RoPE module. Add it now so Phase 4 (full ZayaAttention forward) can use it without further structural changes.

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

- [ ] **Step 1: Add `self.rope` to ZayaAttention.__init__**

Modify the `ZayaAttention` class. Replace its `__init__` with:

```python
class ZayaAttention(nn.Module):
    """Wraps CCA + standard scaled dot product attention.

    Per modular_zaya.py:524-656. Skeleton only — forward implemented in Phase 4.
    """

    def __init__(self, args: ModelArgs, layer_number: int):
        super().__init__()
        self.qkv = CCA(args, layer_number)
        # o_proj input dim is hidden_size // 2 because CCA produces only 8
        # effective query heads (cca_num_q_heads), so the post-attention flat
        # dim is 8 * head_dim = 1024 = hidden_size // 2.
        self.o_proj = nn.Linear(
            args.hidden_size // 2,
            args.hidden_size,
            bias=args.attention_bias,
        )
        # Partial RoPE: rotates first 64 of 128 head dims; traditional=False
        # matches modular_zaya.py's apply_rotary_pos_emb (NeoX style).
        rotary_dim = int((args.hidden_size // args.num_attention_heads) * args.partial_rotary_factor)
        self.rope = nn.RoPE(
            dims=rotary_dim,
            base=args.rope_theta,
            traditional=False,
        )
```

- [ ] **Step 2: Verify weight loading still passes**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_weight_loading.py -v
```
Expected: 8/8 pass. (The new `nn.RoPE` has no learnable parameters, so the param count and weight key matching are unchanged.)

- [ ] **Step 3: Verify partial RoPE tests still pass**

Run:
```bash
.venv/bin/python -m pytest test_partial_rope.py -v
```
Expected: 3/3 pass.

- [ ] **Step 4: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 2 task 3: add nn.RoPE(dims=64, base=5e6, traditional=False) to ZayaAttention"
```

---

### Task 4: Push fork + update STATUS

**Files:**
- Modify: `~/code/personal/zaya1-mlx/STATUS.md`

- [ ] **Step 1: Push the mlx-lm fork**

```bash
cd ~/code/personal/mlx-lm
git push origin zaya1
```

- [ ] **Step 2: Update STATUS.md**

Replace the "Current phase" + "What's done" + "What's next" sections of `~/code/personal/zaya1-mlx/STATUS.md`:

- "Current phase" → `Phase 3 — CCA forward (not yet started).`
- Append to "What's done":
  ```
  Phase 2 (partial RoPE):
    - dump_activations.py augmented to capture 2-tuples (cos, sin) as _0/_1
    - Reference dumps refreshed (smoke, reasoning_short, long_context_seed)
    - 3/3 partial RoPE parity tests pass: synthetic input, dumped cos/sin
      reproducibility, and pre-RoPE Q from reference dump
    - nn.RoPE(dims=64, base=5e6, traditional=False) added to ZayaAttention skeleton
    - Confirmed: mlx-lm's built-in nn.RoPE handles Zaya's partial RoPE natively;
      no custom partial_rope helper needed
  ```
- "What's next" → describe Phase 3 (CCA forward implementation) and that a new plan needs to be written.

- [ ] **Step 3: Commit and push**

```bash
cd ~/code/personal/zaya1-mlx
git add STATUS.md
git commit -m "Phase 2 complete: status update"
git push origin main
```

---

## Phase 2 Gate Verification

- [ ] `cd ~/code/personal/zaya1-mlx/validation && .venv/bin/python -m pytest -q` shows all tests passing (Phase 1's 8 + Phase 2's 3 = 11 total).
- [ ] `~/code/personal/zaya1-mlx/reference/activations/smoke/global_model_rotary_emb_0.npy` and `_1.npy` exist (cos and sin separately).
- [ ] `cd ~/code/personal/mlx-lm && git log origin/zaya1 --oneline | head -3` shows the Phase 2 commit pushed.
- [ ] `STATUS.md` reflects Phase 2 complete.
- [ ] Phase 0 dump_activations tests still pass: `cd ~/code/personal/zaya1-mlx/reference && .venv/bin/python -m pytest -q` shows 5 passed.

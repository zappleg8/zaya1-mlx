# Phase 5: ZayaDecoderATTLayer Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ZayaDecoderATTLayer.__call__` — wires `ResidualScaling` (already has its forward from Phase 1) and `ZayaAttention` (forward from Phase 4) into the full ATT decoder layer that threads the residual stream. Verify parity against PyTorch's `L0_layer_out` reference.

**Architecture:** Per `modular_zaya.py:909-1000`. Order: (1) `res_scale(residual, hidden_states)` if `scale_residual_merge`, (2) initialize `residual = hidden_states.to(fp32)` if it was `None`, else `residual = hidden_states + residual`, (3) `hidden_states = input_norm(residual)` downcast to the norm's dtype, (4) `hidden_states = self_attn(hidden_states, ...)`, (5) return `(hidden_states,), residual, prev_router_hidden_states` — `prev_router_hidden_states` passes through unchanged (ATT layers don't touch it).

The Phase 5 parity test focuses on **layer 0** because residual is `None` there — testable end-to-end from `global_model_embed_tokens_out`. Higher ATT layers can only be tested once Phase 6+7 (MoE forward) exist to compute the intermediate residual state. We add a synthetic-input test for the "non-first layer" code path so both branches are covered.

**Tech Stack:** MLX 0.31.2, mlx-lm fork. No new dependencies.

---

## Pre-flight Setup

- [ ] **Confirm Phase 4 state**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation && \
  .venv/bin/python -m pytest -q 2>&1 | tail -2
```
Expected: 17 passed.

- [ ] **Confirm reference tensors exist**

Run:
```bash
ls ~/code/personal/zaya1-mlx/reference/activations/smoke/global_model_embed_tokens_out.npy && \
  ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L0_layer_out.npy
```
Expected: both exist.

---

### Task 1: Failing test for ZayaDecoderATTLayer

**Files:**
- Create: `~/code/personal/zaya1-mlx/validation/test_att_layer_forward.py`

- [ ] **Step 1: Write the test**

Write `validation/test_att_layer_forward.py`:

```python
"""Phase 5 gate test: ZayaDecoderATTLayer forward parity.

Tests:
  - Layer 0 end-to-end: feed embed_tokens output through layer 0 with
    residual=None, compare layer output to L0_layer_out reference.
  - Synthetic non-first-layer: feed (hidden_states, residual) where
    residual is non-None, verify the residual addition + input_norm
    path runs without error and produces sensible shapes.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"


# Layer-level parity inherits the attention noise floor plus the residual
# path's bf16-vs-fp32 mixing. Generous tolerance at L0 boundary.
LAYER_OUT_TOL = 1e0


def _load_npy(name: str) -> mx.array:
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"Reference tensor {name} missing")
    return mx.array(np.load(path))


def test_att_layer_0_end_to_end(loaded_model):
    """Layer 0: feed embed output with residual=None, compare layer output."""
    embed_out = _load_npy("global_model_embed_tokens_out.npy")
    layer = loaded_model.layers[0]
    weight_dtype = layer.self_attn.qkv.linear_q.weight.dtype
    hs = embed_out.astype(weight_dtype)

    layer_outputs, residual_out, _ = layer(
        hidden_states=hs,
        residual=None,
        mask="causal",
        cache=None,
        prev_router_hidden_states=None,
    )
    layer_out = layer_outputs[0]

    ref = _load_npy("L0_layer_out.npy")
    assert layer_out.shape == ref.shape, (
        f"shape mismatch: mlx={layer_out.shape}, ref={ref.shape}"
    )
    diff = float(mx.max(mx.abs(layer_out.astype(mx.float32) - ref)))
    assert diff < LAYER_OUT_TOL, f"L0 layer_out max abs diff: {diff} (tol {LAYER_OUT_TOL})"


def test_att_layer_0_residual_initialized_in_fp32(loaded_model):
    """When residual is None at layer 0, the returned residual must be fp32."""
    embed_out = _load_npy("global_model_embed_tokens_out.npy")
    layer = loaded_model.layers[0]
    weight_dtype = layer.self_attn.qkv.linear_q.weight.dtype
    hs = embed_out.astype(weight_dtype)

    _, residual_out, _ = layer(
        hidden_states=hs,
        residual=None,
        mask="causal",
        cache=None,
        prev_router_hidden_states=None,
    )
    assert residual_out.dtype == mx.float32, (
        f"residual should be fp32 when residual_in_fp32=True; got {residual_out.dtype}"
    )


def test_att_layer_non_first_residual_path(loaded_model):
    """Even layer != 0 with non-None residual: verify the merge path runs."""
    # Use layer 2 (also ATT) with a synthetic input matching realistic shapes.
    layer = loaded_model.layers[2]
    B, S, H = 1, 7, 2048
    weight_dtype = layer.self_attn.qkv.linear_q.weight.dtype
    hs = mx.zeros((B, S, H)).astype(weight_dtype)
    residual = mx.ones((B, S, H), dtype=mx.float32)

    layer_outputs, residual_out, _ = layer(
        hidden_states=hs,
        residual=residual,
        mask="causal",
        cache=None,
        prev_router_hidden_states=None,
    )
    assert layer_outputs[0].shape == (B, S, H)
    assert residual_out.shape == (B, S, H)
    # The returned residual should differ from the input (since hidden_states != 0 effectively after norm/attn paths
    # AND res_scale's affine alters it). At minimum it must be finite.
    assert mx.all(mx.isfinite(residual_out)).item(), "residual_out has non-finite values"
```

- [ ] **Step 2: Confirm tests fail (no __call__ on ZayaDecoderATTLayer)**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_att_layer_forward.py -v --tb=short 2>&1 | tail -10
```
Expected: tests fail because `ZayaDecoderATTLayer` is not callable.

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add validation/test_att_layer_forward.py
git commit -m "Phase 5 task 1: failing ZayaDecoderATTLayer forward parity tests"
```

---

### Task 2: Implement ZayaDecoderATTLayer.__call__

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

- [ ] **Step 1: Add `__call__` to ZayaDecoderATTLayer**

Add this method inside the `ZayaDecoderATTLayer` class:

```python
    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        prev_router_hidden_states: Optional[mx.array] = None,
        cca_mask: Optional[mx.array] = None,
    ):
        """Even-layer decoder forward (CCA attention).

        Per modular_zaya.py:928-1000.

        Returns:
            outputs: (hidden_states,) — single-element tuple, matching the
                PyTorch convention for layer outputs.
            residual: the updated residual stream (fp32 when residual_in_fp32).
            prev_router_hidden_states: passed through unchanged (ATT layers
                don't run a router).
        """
        # 1. Optional ResidualScaling affine on both streams.
        if hasattr(self, "res_scale"):
            residual, hidden_states = self.res_scale(residual, hidden_states)

        # 2. Initialize residual on the first layer, or accumulate.
        if residual is None:
            residual = hidden_states.astype(mx.float32)
        else:
            residual = hidden_states + residual

        # 3. Pre-norm based on the accumulated residual (downcast to the
        #    norm's weight dtype, typically bf16).
        norm_dtype = self.input_norm.weight.dtype
        hidden_states = self.input_norm(residual.astype(norm_dtype))

        # 4. Attention block.
        hidden_states = self.self_attn(
            hidden_states, mask=mask, cache=cache, cca_mask=cca_mask
        )

        return (hidden_states,), residual, prev_router_hidden_states
```

- [ ] **Step 2: Run the test suite**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_att_layer_forward.py -v --tb=short 2>&1 | tail -10
```
Expected: 3 passed (or 2/3 with the layer_0 diff close to but under LAYER_OUT_TOL = 1e0). If `test_att_layer_0_end_to_end` fails:
- Check shape of `embed_out` — should be (1, 7, 2048).
- Check that the residual fp32 cast happens before input_norm downcast.
- Inspect the layer_out vs ref diff: if it's larger than expected, the bf16 noise from CCA + normalization may compound; tighten only after the full forward (Phase 8).

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 5 task 2: ZayaDecoderATTLayer.__call__ forward"
```

---

### Task 3: Run full validation suite

- [ ] **Step 1: Confirm no regressions**

Run:
```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest -q
```
Expected: 20 passed (8 weight loading + 3 partial RoPE + 3 CCA + 3 attention + 3 ATT layer).

---

### Task 4: Push fork + update STATUS

- [ ] **Step 1: Push the mlx-lm fork**

```bash
cd ~/code/personal/mlx-lm
git push origin zaya1
```

- [ ] **Step 2: Update STATUS.md**

Append "Phase 5 (ATT decoder layer)" to the "What's done" section and change "Current phase" to Phase 6.

- [ ] **Step 3: Commit and push zaya1-mlx**

```bash
cd ~/code/personal/zaya1-mlx
git add STATUS.md
git commit -m "Phase 5 complete: status update"
git push origin main
```

---

## Phase 5 Gate Verification

- [ ] Full validation suite: 20 passed.
- [ ] mlx-lm fork pushed.
- [ ] STATUS reflects Phase 5 complete.
- [ ] Phase 0 dump tests still pass: 5 passed.

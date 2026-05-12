# Phase 7: ZayaDecoderMLPLayer Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ZayaDecoderMLPLayer.__call__` — wraps `ZayaBlock` (Phase 6) in the same residual stream + input_norm machinery as Phase 5's ATT layer, plus threads `prev_router_hidden_states` for the EDA chain. Verify parity at L1 (first MoE layer).

**Architecture:** Per `modular_zaya.py:1425-1533`. Same residual stream logic as `ZayaDecoderATTLayer`, but the block is `ZayaBlock` (which calls the router and updates `prev_router_hidden_states`). `add_bias_linear=False` per config so the bias-add branch is dead code we skip.

**Tech Stack:** MLX 0.31.2. No new dependencies.

---

## Pre-flight Setup

- [ ] Verify Phase 6 (24/24 tests passing) — confirmed already.
- [ ] Required references: `L0_self_attn_out.npy`, `L0_layer_out.npy`, `L1_zaya_block_0.npy`, `L1_zaya_block_1.npy`, `L1_layer_out.npy`.

---

### Task 1: Failing test

**Files:**
- Create: `~/code/personal/zaya1-mlx/validation/test_moe_layer_forward.py`

- [ ] **Step 1: Write test**

`validation/test_moe_layer_forward.py`:

```python
"""Phase 7 gate test: ZayaDecoderMLPLayer (MoE) forward parity at L1.

L1 receives:
  - hidden_states from layer 0 = L0_self_attn_out (the ATT layer's hidden output)
  - residual from layer 0 = L0_layer_out (the residual after layer 0; fp32)
  - prev_router_hidden_states = None (L1 is the first MoE layer)

The dump's 3-tuple handling means L1_layer_out captures the residual
(not hidden_states). L1_zaya_block_0 is the block output (= layer hidden_states
output). L1_zaya_block_1 is the router_hidden_states_next (the EDA chain
input to L3).
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"


LAYER_HS_TOL = 5e-1
RESIDUAL_TOL = 1e-1
ROUTER_HS_TOL = 1e-1


def _load_npy(name: str) -> mx.array:
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"Reference tensor {name} missing")
    return mx.array(np.load(path))


def test_moe_layer_l1_end_to_end(loaded_model):
    """L1 (MoE) forward — hidden_states output should match L1_zaya_block_0."""
    hs_input = _load_npy("L0_self_attn_out.npy")  # hidden_states from L0
    residual_input = _load_npy("L0_layer_out.npy")  # residual after L0 (fp32)
    layer = loaded_model.layers[1]
    weight_dtype = layer.zaya_block.router.down_proj.weight.dtype
    hs = hs_input.astype(weight_dtype)

    layer_outputs, residual_out, router_hs_out = layer(
        hidden_states=hs,
        residual=residual_input,
        prev_router_hidden_states=None,
    )
    layer_hs = layer_outputs[0]

    ref = _load_npy("L1_zaya_block_0.npy")
    assert layer_hs.shape == ref.shape
    diff = float(mx.max(mx.abs(layer_hs.astype(mx.float32) - ref)))
    assert diff < LAYER_HS_TOL, f"L1 layer hidden_states diff: {diff}"


def test_moe_layer_l1_residual_output(loaded_model):
    """Residual after L1 should match L1_layer_out (fp32)."""
    hs_input = _load_npy("L0_self_attn_out.npy")
    residual_input = _load_npy("L0_layer_out.npy")
    layer = loaded_model.layers[1]
    weight_dtype = layer.zaya_block.router.down_proj.weight.dtype
    hs = hs_input.astype(weight_dtype)

    _, residual_out, _ = layer(
        hidden_states=hs,
        residual=residual_input,
        prev_router_hidden_states=None,
    )

    ref = _load_npy("L1_layer_out.npy")
    assert residual_out.shape == ref.shape
    diff = float(mx.max(mx.abs(residual_out - ref)))
    assert diff < RESIDUAL_TOL, f"L1 residual_out diff: {diff}"


def test_moe_layer_l1_router_hs_output(loaded_model):
    """router_hidden_states_next should match L1_zaya_block_1 (feeds L3's EDA)."""
    hs_input = _load_npy("L0_self_attn_out.npy")
    residual_input = _load_npy("L0_layer_out.npy")
    layer = loaded_model.layers[1]
    weight_dtype = layer.zaya_block.router.down_proj.weight.dtype
    hs = hs_input.astype(weight_dtype)

    _, _, router_hs_out = layer(
        hidden_states=hs,
        residual=residual_input,
        prev_router_hidden_states=None,
    )

    ref = _load_npy("L1_zaya_block_1.npy")
    assert router_hs_out.shape == ref.shape
    diff = float(mx.max(mx.abs(router_hs_out.astype(mx.float32) - ref)))
    assert diff < ROUTER_HS_TOL, f"L1 router_hs_out diff: {diff}"
```

- [ ] **Step 2: Confirm failure**: `pytest test_moe_layer_forward.py -v`

- [ ] **Step 3: Commit failing test**

---

### Task 2: Implement ZayaDecoderMLPLayer.__call__

Add `__call__` to ZayaDecoderMLPLayer (the structure mirrors ATT decoder layer; differences highlighted below):

```python
    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array] = None,
        mask=None,
        cache=None,
        prev_router_hidden_states: Optional[mx.array] = None,
        cca_mask: Optional[mx.array] = None,
    ):
        """MoE decoder forward.

        Per modular_zaya.py:1459-1533. Same residual+norm pattern as ATT,
        but uses zaya_block and threads prev_router_hidden_states.
        """
        if hasattr(self, "res_scale"):
            residual, hidden_states = self.res_scale(residual, hidden_states)

        if residual is None:
            residual = hidden_states.astype(mx.float32)
        else:
            residual = hidden_states + residual

        norm_dtype = self.input_norm.weight.dtype
        hidden_states = self.input_norm(residual.astype(norm_dtype))

        # add_bias_linear=False per config; we always go through this branch.
        hidden_states, _bias, prev_router_hidden_states = self.zaya_block(
            hidden_states, prev_router_hidden_states=prev_router_hidden_states
        )

        return (hidden_states,), residual, prev_router_hidden_states
```

---

### Task 3: Run full suite — expected 27 passed.

### Task 4: Push fork + STATUS update.

---

## Phase 7 Gate Verification

- [ ] Full validation suite: 27 passed.
- [ ] mlx-lm fork pushed.
- [ ] STATUS reflects Phase 7 complete.

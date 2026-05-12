# Phase 6: MoE Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ZayaRouter.__call__` and `ZayaBlock.__call__` — the MoE forward path with EDA + top-1 routing + balancing biases + MoD skip-expert. Verify parity against PyTorch reference dumps at L1 (first MoE layer, no EDA).

**Architecture:** Per `modular_zaya.py:1036-1422`. The forward is:

1. **Router** (`ZayaRouter`):
   - `hs = down_proj(hidden_states)` → (B, S, 256)
   - If EDA: `hs = hs + router_states * router_states_scale`
   - Save `hs` as `router_hidden_states_next` for the next MoE layer
   - `hs_norm = rmsnorm_eda(hs)`
   - `logits = router_mlp(hs_norm)` → (B, S, 17)
   - `probs = softmax(logits, dim=-1)`
   - `biased = probs.detach().to(fp32) + balancing_biases`
   - `expert_choice = topk(biased, 1).indices`
   - `route_prob = gather(probs, dim=-1, index=expert_choice)`
   - Return `(route_prob_flat, expert_choice_flat, router_hidden_states_next)`

2. **ZayaBlock**:
   - Call router → (probs, indices, next_router_states)
   - Flatten hidden_states to (B*S, H), indices to (B*S,)
   - Sort indices: `sorted_indices, sort_order = sort(indices)`
   - Count tokens per expert (one-hot + sum)
   - `sorted_hs = hidden_states[sort_order]`
   - `original_order = argsort(sort_order)`
   - **MoD branch**: only run the first 16 (real) experts on their slices; tokens routed to skip expert (index 16) pass through unchanged
   - `expert_output[original_order]` to un-permute
   - Reshape to (B, S, H)
   - `expert_output * route_prob.unsqueeze(-1)`

The hardest piece is the MoD permute/un-permute logic. We test at L1 (first MoE layer, `use_eda=False`) because the EDA path requires a `prev_router_states` that only exists once a prior MoE layer has run — Phase 8 will exercise the EDA chain end-to-end.

**Tech Stack:** MLX 0.31.2, mlx-lm fork. MLX has `mx.argsort`, `mx.sort`, fancy-indexing gather. No native `bincount` — we use one-hot + sum.

---

## Pre-flight Setup

- [ ] **Confirm Phase 5 state**

```bash
cd ~/code/personal/zaya1-mlx/validation && \
  .venv/bin/python -m pytest -q 2>&1 | tail -2
```
Expected: 21 passed.

- [ ] **Confirm L1 reference tensors exist**

```bash
ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L1_input_norm_out.npy && \
  ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L1_zaya_block_router_router_mlp_4_out.npy && \
  ls ~/code/personal/zaya1-mlx/reference/activations/smoke/L1_zaya_block_out.npy
```
Expected: all three exist.

---

### Task 1: Failing tests for Router and ZayaBlock

**Files:**
- Create: `~/code/personal/zaya1-mlx/validation/test_moe_forward.py`

- [ ] **Step 1: Write the test**

Write `validation/test_moe_forward.py`:

```python
"""Phase 6 gate test: ZayaRouter and ZayaBlock forward parity at L1.

L1 is the first MoE layer in the network (zaya_first_layer=1). EDA is gated
off there — `use_eda=False`. We feed L1_input_norm_out (Phase 0 dump) directly
into model.layers[1].zaya_block and compare:
  - Router MLP logits (mlp_4_out): the pre-softmax expert logits
  - Final block output: the MoE block's output (post-gate)

Tolerances are bf16-aware (the router involves a softmax + GELU + multiple
matmuls, each with bf16 noise propagation).
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"


# bf16 noise budgets for MoE intermediates.
ROUTER_LOGITS_TOL = 2e-1
ROUTER_PROB_TOL = 1e-1
BLOCK_OUT_TOL = 5e-1


def _load_npy(name: str) -> mx.array:
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"Reference tensor {name} missing")
    return mx.array(np.load(path))


def test_router_l1_logits(loaded_model):
    """L1 router MLP should produce logits matching the reference."""
    hs = _load_npy("L1_input_norm_out.npy")
    block = loaded_model.layers[1].zaya_block
    weight_dtype = block.router.down_proj.weight.dtype
    hs = hs.astype(weight_dtype)

    route_prob, expert_choice, router_states_next = block.router(
        hs, router_states=None
    )

    # The reference router_mlp_4_out is the pre-softmax logits.
    # Our router doesn't expose them directly — we reconstruct by walking
    # the same path in isolation.
    h = block.router.down_proj(hs)
    # No EDA at layer 1
    h_norm = block.router.rmsnorm_eda(h)
    logits = block.router.router_mlp(h_norm)

    ref = _load_npy("L1_zaya_block_router_router_mlp_4_out.npy")
    diff = float(mx.max(mx.abs(logits.astype(mx.float32) - ref)))
    assert diff < ROUTER_LOGITS_TOL, f"L1 router logits max abs diff: {diff}"


def test_router_l1_top1_choices_match(loaded_model):
    """At L1, our router's top-1 choices should match the reference's.

    The reference saves `route_prob_flat` (B*S, 1) as `_out` — we can derive
    expert_choice by computing it from the logits and matching argmax."""
    hs = _load_npy("L1_input_norm_out.npy")
    block = loaded_model.layers[1].zaya_block
    weight_dtype = block.router.down_proj.weight.dtype
    hs = hs.astype(weight_dtype)

    route_prob_mlx, expert_choice_mlx, _ = block.router(hs, router_states=None)

    # Compute the reference logits via the same path
    ref_logits = _load_npy("L1_zaya_block_router_router_mlp_4_out.npy")
    ref_probs = mx.softmax(ref_logits.astype(mx.float32), axis=-1)
    ref_biased = ref_probs + block.router.balancing_biases.astype(mx.float32)
    ref_top1 = mx.argmax(ref_biased, axis=-1)  # (B, S)
    ref_top1_flat = ref_top1.reshape(-1)

    assert expert_choice_mlx.shape == (ref_top1_flat.size, 1), (
        f"shape mismatch: {expert_choice_mlx.shape}"
    )
    mlx_top1 = expert_choice_mlx.reshape(-1)
    matches = int(mx.sum(mlx_top1 == ref_top1_flat).item())
    total = mlx_top1.size
    assert matches == total, (
        f"top-1 choice mismatch: {matches}/{total} match. "
        f"This indicates routing diverges at one or more positions."
    )


def test_zaya_block_l1_output(loaded_model):
    """L1 ZayaBlock end-to-end output should match the reference."""
    hs = _load_npy("L1_input_norm_out.npy")
    block = loaded_model.layers[1].zaya_block
    weight_dtype = block.router.down_proj.weight.dtype
    hs = hs.astype(weight_dtype)

    out, _bias, _next_router_hs = block(
        hidden_states=hs, prev_router_hidden_states=None
    )

    ref = _load_npy("L1_zaya_block_out.npy")
    assert out.shape == ref.shape, f"shape: mlx={out.shape}, ref={ref.shape}"
    diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref)))
    assert diff < BLOCK_OUT_TOL, f"L1 zaya_block_out max abs diff: {diff}"
```

- [ ] **Step 2: Run and verify tests fail (Router not callable, ZayaBlock not callable)**

```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_moe_forward.py -v --tb=short 2>&1 | tail -10
```
Expected: 3 failures, error messages mention `'ZayaRouter' object is not callable` and similar.

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add validation/test_moe_forward.py
git commit -m "Phase 6 task 1: failing MoE forward parity tests"
```

---

### Task 2: Implement ZayaRouter.__call__

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

- [ ] **Step 1: Add `__call__` to ZayaRouter**

Add inside `ZayaRouter` class (after `__init__`):

```python
    def __call__(
        self,
        hidden_states: mx.array,
        router_states: Optional[mx.array] = None,
    ):
        """Router forward.

        Args:
          hidden_states: (B, S, H) — the layer's pre-router input.
          router_states: (B, S, mlp_expansion=256) from the previous MoE
            layer's router. None at the first MoE layer.

        Returns:
          route_prob: (B*S, topk=1) gathered probabilities for chosen experts.
          expert_choice: (B*S, topk=1) chosen expert indices.
          router_hidden_states_next: (B, S, mlp_expansion) the pre-norm
            post-EDA hs, to feed the next MoE layer's router (EDA chain).
        """
        B, S, _ = hidden_states.shape
        hs = self.down_proj(hidden_states)  # (B, S, mlp_expansion=256)

        if self.use_eda and router_states is not None:
            hs = hs + router_states * self.router_states_scale

        # Stash pre-norm post-EDA hs for the next router.
        router_hidden_states_next = hs

        # Normalize, then MLP to expert logits.
        hs_norm = self.rmsnorm_eda(hs)
        logits = self.router_mlp(hs_norm)  # (B, S, num_experts)

        # Expert probabilities (in input dtype) and selection (in fp32 to
        # match PyTorch's `probs.detach().to(torch.float32) + biases`).
        expert_prob = mx.softmax(logits, axis=-1)
        biased = expert_prob.astype(mx.float32) + self.balancing_biases.astype(mx.float32)

        # Top-1 selection. argmax returns int indices; reshape to (B, S, 1)
        # to match the topk=1 shape convention.
        expert_choice = mx.argmax(biased, axis=-1, keepdims=True)  # (B, S, 1)

        # Gather the chosen expert's probability.
        route_prob = mx.take_along_axis(expert_prob, expert_choice, axis=-1)

        # Flatten the batch/sequence dims; topk dim retained.
        route_prob_flat = route_prob.reshape(-1, 1)
        expert_choice_flat = expert_choice.reshape(-1, 1)

        return route_prob_flat, expert_choice_flat, router_hidden_states_next
```

- [ ] **Step 2: Run the logits test to confirm router path works**

```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_moe_forward.py::test_router_l1_logits -v --tb=short 2>&1 | tail -8
```
Expected: PASS (the test reconstructs logits manually via the same submodules; our router's internals match).

- [ ] **Step 3: Run the top-1 match test**

```bash
.venv/bin/python -m pytest test_moe_forward.py::test_router_l1_top1_choices_match -v --tb=short 2>&1 | tail -8
```
Expected: PASS. If not all 7 tokens match, the router has a bug.

- [ ] **Step 4: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 6 task 2: ZayaRouter.__call__ forward"
```

---

### Task 3: Implement ZayaBlock.__call__

**Files:**
- Modify: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

- [ ] **Step 1: Add `__call__` to ZayaBlock**

Add inside `ZayaBlock` class:

```python
    def __call__(
        self,
        hidden_states: mx.array,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        """MoE block forward.

        Args:
          hidden_states: (B, S, H) — pre-MoE hidden state.
          prev_router_hidden_states: (B, S, mlp_expansion) for the EDA chain;
            None at the first MoE layer.

        Returns:
          expert_output: (B, S, H) — gated MoE output.
          mlp_bias: None (we do not run add_bias_linear paths in this port).
          router_hidden_states_next: (B, S, mlp_expansion) for the next
            MoE layer's EDA.
        """
        B, S, H = hidden_states.shape
        route_prob, expert_choice, router_hidden_states_next = self.router(
            hidden_states, router_states=prev_router_hidden_states
        )

        # Flatten batch and sequence for routing.
        hidden_flat = hidden_states.reshape(B * S, H)
        # expert_choice: (B*S, 1); flatten to (B*S,)
        indices_flat = expert_choice.reshape(-1)
        # route_prob: (B*S, 1); flatten to (B*S,)
        probs_flat = route_prob.reshape(-1)

        # Sort tokens by their assigned expert.
        sort_order = mx.argsort(indices_flat)
        sorted_indices = indices_flat[sort_order]
        sorted_hidden = hidden_flat[sort_order]

        # Tokens per expert (no native bincount; one-hot + sum).
        num_experts = self.router.num_experts  # 17 with MoD
        expert_ids = mx.arange(num_experts)
        one_hot = mx.equal(sorted_indices[:, None], expert_ids[None, :])
        tokens_per_expert = mx.sum(one_hot, axis=0).astype(mx.int32)

        # Run each real expert on its slice of sorted_hidden.
        # MoD: tokens routed to the skip expert (index num_experts - 1) are
        # passed through unchanged.
        num_real_experts = num_experts - 1 if self.use_mod else num_experts
        real_chunks = []
        cursor = 0
        for e in range(num_real_experts):
            count = int(tokens_per_expert[e].item())
            if count == 0:
                continue
            chunk = sorted_hidden[cursor : cursor + count]
            real_chunks.append(self.experts.local_experts[e](chunk))
            cursor += count

        if self.use_mod:
            # Skip expert: pass through unchanged.
            skip_count = int(tokens_per_expert[num_real_experts].item())
            if skip_count > 0:
                real_chunks.append(sorted_hidden[cursor : cursor + skip_count])

        expert_output_sorted = (
            mx.concatenate(real_chunks, axis=0)
            if real_chunks
            else mx.zeros_like(sorted_hidden)
        )

        # Un-permute back to original token order.
        original_order = mx.argsort(sort_order)
        expert_output = expert_output_sorted[original_order]
        expert_output = expert_output.reshape(B, S, H)

        # Scale each token's output by its routing probability.
        expert_output = expert_output * probs_flat.reshape(B, S, 1)

        return expert_output, None, router_hidden_states_next
```

- [ ] **Step 2: Run the block output test**

```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest test_moe_forward.py::test_zaya_block_l1_output -v --tb=short 2>&1 | tail -10
```
Expected: PASS within `BLOCK_OUT_TOL = 5e-1`. If diff is much larger:
- Check that `route_prob` is being applied with broadcasting on the right axis.
- Check that `original_order` correctly inverts `sort_order`: `flat[sort_order][original_order]` should equal `flat`.
- Check that `MLP.__call__` matches the SwiGLU semantics (already validated as a primitive, but worth double-checking that the test reproduces a single expert's output if all tokens routed to it).

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/mlx-lm
git add mlx_lm/models/zaya.py
git commit -m "Phase 6 task 3: ZayaBlock.__call__ forward with MoD routing"
```

---

### Task 4: Full suite

```bash
cd ~/code/personal/zaya1-mlx/validation
.venv/bin/python -m pytest -q
```
Expected: 24 passed (21 prior + 3 MoE).

---

### Task 5: Push fork + STATUS

- [ ] **Step 1: Push the mlx-lm fork**

```bash
cd ~/code/personal/mlx-lm
git push origin zaya1
```

- [ ] **Step 2: Update STATUS.md**

Append "Phase 6 (MoE forward)" to "What's done" and change "Current phase" to Phase 7.

- [ ] **Step 3: Push zaya1-mlx**

```bash
cd ~/code/personal/zaya1-mlx
git add STATUS.md
git commit -m "Phase 6 complete: status update"
git push origin main
```

---

## Phase 6 Gate Verification

- [ ] Full validation suite: 24 passed.
- [ ] mlx-lm fork pushed.
- [ ] STATUS reflects Phase 6 complete.

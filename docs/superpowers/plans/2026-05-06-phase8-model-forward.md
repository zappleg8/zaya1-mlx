# Phase 8: ZayaModel Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `ZayaModel.__call__` — wires all 80 alternating decoder layers + embed_tokens + final ResidualScaling + final RMSNorm. Threads residual + prev_router_hidden_states through the layer stack. Verify end-to-end pre-lm_head parity against `global_model_final_norm_out`.

**Architecture:** Per `modular_zaya.py:1700-1863`. Forward in MLX HF layout:

1. `h = embed_tokens(input_ids)` or use `input_embeddings` if provided.
2. Initialize `residual = None`, `prev_router_hidden_states = None`.
3. For each layer 0..79: call layer with `(hs, residual, mask="causal", prev_router_hidden_states)`. Update all three.
4. Final residual merge: `if scale_residual_merge: residual, h = res_scale(residual, h); residual = h + residual`.
5. `h = final_norm(residual.astype(norm_dtype))`.
6. Return `h` (the pre-lm_head hidden state).

Tolerance is generous (~2e0) — bf16 noise compounds through 80 layers, but each layer normalizes so it doesn't multiply unboundedly.

---

### Task 1: Failing test

`validation/test_model_forward.py`:

```python
from pathlib import Path
import mlx.core as mx
import numpy as np
import pytest

REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"
MODEL_OUT_TOL = 2e0

def _load_npy(name):
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"Reference tensor {name} missing")
    return mx.array(np.load(path))


def test_full_model_forward(loaded_model):
    """End-to-end ZayaModel.forward parity. Feeds the captured embed_tokens
    output through all 80 layers and compares the final_norm output."""
    embed_out = _load_npy("global_model_embed_tokens_out.npy")
    h = loaded_model.model(inputs=None, cache=None, input_embeddings=embed_out)
    ref = _load_npy("global_model_final_norm_out.npy")
    assert h.shape == ref.shape, f"shape: mlx={h.shape}, ref={ref.shape}"
    diff = float(mx.max(mx.abs(h.astype(mx.float32) - ref)))
    assert diff < MODEL_OUT_TOL, f"final_norm output max abs diff: {diff}"
```

### Task 2: Implement ZayaModel.__call__

```python
    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        residual = None
        prev_router_hs = None

        for layer in self.layers:
            outputs, residual, prev_router_hs = layer(
                hidden_states=h,
                residual=residual,
                mask="causal",
                cache=None,  # per-layer cache support comes with Phase 10 generation
                prev_router_hidden_states=prev_router_hs,
            )
            h = outputs[0]

        # Final residual merge.
        if hasattr(self, "res_scale"):
            residual, h = self.res_scale(residual, h)
        if residual is None:
            residual = h.astype(mx.float32)
        else:
            residual = h + residual

        norm_dtype = self.final_norm.weight.dtype
        return self.final_norm(residual.astype(norm_dtype))
```

### Task 3: Verify suite passes (28/28 expected).

### Task 4: Push + STATUS update.

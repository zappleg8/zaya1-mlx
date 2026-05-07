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
    hs = _load_npy(f"L{layer_idx}_input_norm_out.npy")
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

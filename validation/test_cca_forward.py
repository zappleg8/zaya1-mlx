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

    hs = _load_npy(f"L{layer_idx}_input_norm_out.npy")  # (B, S, H)
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

"""Phase 7 gate test: ZayaDecoderMLPLayer (MoE) forward parity at L1.

L1 receives:
  - hidden_states from layer 0 = L0_self_attn_out
  - residual from layer 0 = L0_layer_out (fp32)
  - prev_router_hidden_states = None
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
    hs_input = _load_npy("L0_self_attn_out.npy")
    residual_input = _load_npy("L0_layer_out.npy")
    layer = loaded_model.layers[1]
    weight_dtype = layer.zaya_block.router.down_proj.weight.dtype
    hs = hs_input.astype(weight_dtype)

    layer_outputs, _residual_out, _router_hs_out = layer(
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

"""Phase 6 gate test: ZayaRouter and ZayaBlock forward parity at L1.

L1 is the first MoE layer in the network (zaya_first_layer=1). EDA is gated
off there — use_eda=False. We feed L1_input_norm_out (Phase 0 dump) directly
into model.layers[1].zaya_block and compare router logits + final output.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"


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

    h = block.router.down_proj(hs)
    h_norm = block.router.rmsnorm_eda(h)
    logits = block.router.router_mlp(h_norm)

    ref = _load_npy("L1_zaya_block_router_router_mlp_4_out.npy")
    diff = float(mx.max(mx.abs(logits.astype(mx.float32) - ref)))
    assert diff < ROUTER_LOGITS_TOL, f"L1 router logits max abs diff: {diff}"


def test_router_l1_top1_choices_match(loaded_model):
    """At L1, our router's top-1 choices should match the reference's."""
    hs = _load_npy("L1_input_norm_out.npy")
    block = loaded_model.layers[1].zaya_block
    weight_dtype = block.router.down_proj.weight.dtype
    hs = hs.astype(weight_dtype)

    route_prob_mlx, expert_choice_mlx, _ = block.router(hs, router_states=None)

    ref_logits = _load_npy("L1_zaya_block_router_router_mlp_4_out.npy")
    ref_probs = mx.softmax(ref_logits.astype(mx.float32), axis=-1)
    ref_biased = ref_probs + block.router.balancing_biases.astype(mx.float32)
    ref_top1 = mx.argmax(ref_biased, axis=-1)
    ref_top1_flat = ref_top1.reshape(-1)

    assert expert_choice_mlx.shape == (ref_top1_flat.size, 1), (
        f"shape mismatch: {expert_choice_mlx.shape}"
    )
    mlx_top1 = expert_choice_mlx.reshape(-1)
    matches = int(mx.sum(mlx_top1 == ref_top1_flat).item())
    total = mlx_top1.size
    assert matches == total, (
        f"top-1 choice mismatch: {matches}/{total} match."
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

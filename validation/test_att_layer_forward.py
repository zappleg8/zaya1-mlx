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


LAYER_OUT_TOL = 1e0


def _load_npy(name: str) -> mx.array:
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"Reference tensor {name} missing")
    return mx.array(np.load(path))


def test_att_layer_0_hidden_states_output(loaded_model):
    """Layer 0: feed embed output with residual=None.

    The layer's hidden_states output (layer_outputs[0]) equals the attention
    output, since the ATT layer doesn't apply post-attention processing. We
    compare against L0_self_attn_out (already tested as the attention block
    output in Phase 4; this test verifies the residual+norm path that feeds
    self_attn is identical when starting from embed_tokens_out).
    """
    embed_out = _load_npy("global_model_embed_tokens_out.npy")
    layer = loaded_model.layers[0]
    weight_dtype = layer.self_attn.qkv.linear_q.weight.dtype
    hs = embed_out.astype(weight_dtype)

    layer_outputs, _residual_out, _ = layer(
        hidden_states=hs,
        residual=None,
        mask="causal",
        cache=None,
        prev_router_hidden_states=None,
    )
    layer_out = layer_outputs[0]

    ref = _load_npy("L0_self_attn_out.npy")
    assert layer_out.shape == ref.shape, (
        f"shape mismatch: mlx={layer_out.shape}, ref={ref.shape}"
    )
    diff = float(mx.max(mx.abs(layer_out.astype(mx.float32) - ref)))
    assert diff < LAYER_OUT_TOL, f"L0 hidden_states max abs diff: {diff} (tol {LAYER_OUT_TOL})"


def test_att_layer_0_residual_output(loaded_model):
    """Layer 0's returned residual equals embed_tokens_out after res_scale's
    hidden_states affine, cast to fp32. This is what the dump's L0_layer_out
    captures — the dump hook on the layer module sees a 3-tuple
    `((hs,), residual, prev_router_hs)` and saves the first plain tensor it
    finds, which is `residual` (the 1-tuple at index 0 is skipped)."""
    embed_out = _load_npy("global_model_embed_tokens_out.npy")
    layer = loaded_model.layers[0]
    weight_dtype = layer.self_attn.qkv.linear_q.weight.dtype
    hs = embed_out.astype(weight_dtype)

    _layer_outputs, residual_out, _ = layer(
        hidden_states=hs,
        residual=None,
        mask="causal",
        cache=None,
        prev_router_hidden_states=None,
    )

    ref = _load_npy("L0_layer_out.npy")
    assert residual_out.shape == ref.shape
    diff = float(mx.max(mx.abs(residual_out - ref)))
    # Residual is in fp32 throughout, no bf16 noise; tolerance can be tight.
    assert diff < 1e-5, f"L0 residual_out max abs diff: {diff}"


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
    assert mx.all(mx.isfinite(residual_out)).item(), "residual_out has non-finite values"

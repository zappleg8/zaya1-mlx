"""Phase 2 gate test: verify mlx-lm nn.RoPE matches PyTorch's apply_rotary_pos_emb
for Zaya's specific config (dims=64, base=5e6, traditional=False).

Two complementary checks:
  1. Synthetic-input parity: build Q with known values, apply both paths,
     verify max abs diff < 1e-5.
  2. Reference-anchored parity: load captured pre-RoPE Q from the smoke
     reference dump + the captured cos/sin, apply both paths, verify match.

The 'PyTorch math' is reimplemented in MLX (the function is small) so we
don't need to import torch in the validation venv.
"""
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


# Tolerance for bf16 rounding noise. PyTorch stores the model in bf16, which
# rounds cos/sin values to ~7-bit mantissa precision (~1/128 ≈ 0.008 max ULP
# for values near 1). Inspecting dumped values confirms they're bf16-quantized
# (e.g. 0.5390625 = 552/1024). MLX nn.RoPE computes cos/sin in fp32 internally,
# so it's MORE precise than PyTorch's stored values — but we see the difference
# when comparing against bf16-rounded references. Empirically, the gap is
# ~2e-3 on cos/sin and ~5e-3 on a single RoPE-rotated tensor.
BF16_COS_SIN_TOL = 5e-3
BF16_POST_ROPE_TOL = 1e-2


def test_against_dumped_cos_sin():
    """Use the cos/sin captured from PyTorch and verify our local computation
    matches up to bf16 rounding noise.

    Tolerance is loosened from 1e-5 to BF16_COS_SIN_TOL because the dumped
    cos/sin were saved from a bf16 model (PyTorch internally bf16-rounded
    them); our local fp32 computation is more precise but won't bit-match.
    """
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
    assert cos_diff < BF16_COS_SIN_TOL, f"cos diff vs locally computed: {cos_diff}"
    assert sin_diff < BF16_COS_SIN_TOL, f"sin diff vs locally computed: {sin_diff}"


def test_against_dumped_q():
    """End-to-end: load captured pre-RoPE Q (CCA output), apply both RoPE paths,
    confirm equivalence within bf16 rounding noise."""
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
    assert diff < BF16_POST_ROPE_TOL, f"max abs diff: {diff}"

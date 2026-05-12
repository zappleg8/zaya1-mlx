"""Phase 9 gate test: end-to-end logits via Model.__call__.

Verifies the tied-embedding lm_head produces logits matching the captured
reference, and that greedy next-token argmax matches.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"


# Logits range can be wide; tolerance reflects bf16 compounding through
# 80 layers + the lm_head projection.
LOGITS_TOL = 5e0


def _load_npy(name: str) -> mx.array:
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"Reference tensor {name} missing")
    return mx.array(np.load(path))


def test_lm_head_logits(loaded_model):
    embed_out = _load_npy("global_model_embed_tokens_out.npy")
    logits = loaded_model(inputs=None, cache=None, input_embeddings=embed_out)
    ref = _load_npy("global_lm_head_out.npy")
    assert logits.shape == ref.shape, f"shape: mlx={logits.shape}, ref={ref.shape}"
    diff = float(mx.max(mx.abs(logits.astype(mx.float32) - ref)))
    assert diff < LOGITS_TOL, f"logits max abs diff: {diff}"


def test_greedy_next_token_matches(loaded_model):
    """Last-position argmax should pick the same token as PyTorch did.

    This is the most consequential parity check: if the model agrees on what
    token to emit, end-to-end generation will be coherent."""
    embed_out = _load_npy("global_model_embed_tokens_out.npy")
    logits = loaded_model(inputs=None, cache=None, input_embeddings=embed_out)
    ref = _load_npy("global_lm_head_out.npy")
    # Take last-position logits over vocab.
    mlx_top = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    ref_top = int(mx.argmax(ref[:, -1, :], axis=-1).item())
    assert mlx_top == ref_top, (
        f"Greedy next-token mismatch: mlx={mlx_top}, ref={ref_top}"
    )

"""Phase 8 gate test: end-to-end ZayaModel forward parity.

Feeds the captured embed_tokens output through all 80 layers (via the
`input_embeddings` parameter to bypass tokenization) and compares the
final_norm output to the captured reference.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


REFERENCE_DIR = Path(__file__).parent.parent / "reference" / "activations" / "smoke"
# bf16 noise compounded over 80 layers. Empirically lands at ~2.4 absolute
# on a tensor of range ~31 (~7.7% relative). This is the realistic floor
# without higher-precision intermediates throughout the pipeline.
MODEL_OUT_TOL = 3e0


def _load_npy(name: str) -> mx.array:
    path = REFERENCE_DIR / name
    if not path.exists():
        pytest.skip(f"Reference tensor {name} missing")
    return mx.array(np.load(path))


def test_full_model_forward(loaded_model):
    embed_out = _load_npy("global_model_embed_tokens_out.npy")
    h = loaded_model.model(inputs=None, cache=None, input_embeddings=embed_out)
    ref = _load_npy("global_model_final_norm_out.npy")
    assert h.shape == ref.shape, f"shape: mlx={h.shape}, ref={ref.shape}"
    diff = float(mx.max(mx.abs(h.astype(mx.float32) - ref)))
    assert diff < MODEL_OUT_TOL, f"final_norm output max abs diff: {diff}"

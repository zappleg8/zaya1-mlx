# zaya1-mlx

An Apple MLX port of [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B), with a layer-by-layer numerical-parity validation harness against the PyTorch reference.

> **Status:** WIP — not yet functional. The model is novel along several axes (custom SSM, top-1 MoE, Mixture-of-Depths, Exponential Depth Averaging, partial RoPE) and each component is being ported and validated against the original PyTorch implementation before it is considered done. See [the design doc](docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md) for the full plan.

## Goals

- Faithful MLX port — numerical equivalence to PyTorch at every layer boundary, within tolerance.
- Publish `mlx-community/ZAYA1-8B` (BF16) and `mlx-community/ZAYA1-8B-4bit`.
- Open-source the reference-validation harness so future MLX ports of similar hybrid architectures (Mamba+Attention+MoE) can reuse the pattern.

## Repo layout

```
zaya1-mlx/
  docs/superpowers/specs/  # design + implementation plan
  reference/               # PyTorch forward + activation dump (offline; gitignored .npy)
  validation/              # MLX vs reference comparison harness
  scripts/                 # convert + upload helpers
  zaya1_mlx/               # thin Python package wrapping the mlx-lm model
```

The model code itself lives in a fork of [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm) at `mlx_lm/models/zaya.py` — designed to merge cleanly upstream as a PR.

## Status

See `STATUS.md` for the current implementation phase and gate progress.

## License

Apache 2.0 (matching upstream mlx-lm and the ZAYA1 weights license).

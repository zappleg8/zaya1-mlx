# CLAUDE.md — zaya1-mlx

This file is the bootstrap for any new context window picking up this project. Read it first.

## What this is

Apple MLX port of `Zyphra/ZAYA1-8B`, with reference-driven numerical-parity validation against PyTorch.

## Where things live

- **This repo** (`~/code/personal/zaya1-mlx`): everything *around* the model — reference harness, validation, HF upload scripts, future agent harness.
- **mlx-lm fork** (`~/code/personal/mlx-lm`): the model code itself, in `mlx_lm/models/zaya.py`. Designed to be PR-ready upstream.
- **Hugging Face cache**: ZAYA1-8B weights expected at `~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B/`.

## Bootstrap reading order

1. [`STATUS.md`](STATUS.md) — current phase, gate progress, blockers.
2. [`docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md`](docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md) — full design.
3. The implementation plan (forthcoming, also in `docs/superpowers/specs/`).

## Working norms

- **Validation gates are non-negotiable.** Each phase has a numerical-parity gate (see design §8). Do not advance without passing.
- **Never load PyTorch and MLX in the same Python process.** RAM constraint on the M3 Max (36 GB). The reference workflow is offline-dump-then-compare; see design §7.
- **Update `STATUS.md` at the end of each work session.** Future sessions read it first.
- **PII discipline:** this repo is public. No personal information, no employer references, no real names in commits or docs. Git authorship uses the GitHub noreply email.

## Key external links

- Model card: https://huggingface.co/Zyphra/ZAYA1-8B
- Zyphra transformers fork (zaya1 branch): https://github.com/Zyphra/transformers/tree/zaya1
- `modeling_zaya.py`: https://github.com/Zyphra/transformers/blob/zaya1/src/transformers/models/zaya/modeling_zaya.py
- `configuration_zaya.py`: https://github.com/Zyphra/transformers/blob/zaya1/src/transformers/models/zaya/configuration_zaya.py
- mlx-lm reference Mamba2: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba2.py
- mlx-lm Jamba (closest hybrid analogue): https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/jamba.py

# Status

**Last updated:** 2026-05-06

## Current phase

**Phase 1 — COMPLETE.** Phase 2 (partial RoPE wrapper) not yet started.

## What's done

Phase 0 (reference scaffolding):
- Reference uv venv at `reference/.venv` with `torch==2.5.1` (CPU) + `transformers==4.57.1` from `Zyphra/transformers @ zaya1` (commit `f0ab5bef`)
- ZAYA1-8B weights downloaded (16 GB)
- Source code read end-to-end; architecture cataloged at [`reference/notes/zaya-architecture.md`](reference/notes/zaya-architecture.md)
- All 5 open architectural questions from spec §5 resolved
- Spec amended (R1): no SSM in ZAYA1; CCA replaces it; layer schedule is 1:1 ATT/MoE alternation
- `dump_activations.py` + 5/5 tests passing
- 3 reference dumps captured (smoke, reasoning_short, long_context_seed; 3,046 tensors each; 746 MB total)

Phase 1 (skeleton + weight loading):
- HF weight key inventory documented at [`reference/notes/hf-weight-keys.md`](reference/notes/hf-weight-keys.md): 28 unique key patterns, 2,483 total tensors verified to match index exactly
- mlx-lm conventions documented at [`reference/notes/mlx-lm-conventions.md`](reference/notes/mlx-lm-conventions.md)
- Validation venv at `validation/.venv` with MLX 0.31.2 + editable `~/code/personal/mlx-lm`
- Skeleton file [`~/code/personal/mlx-lm/mlx_lm/models/zaya.py`](../mlx-lm/mlx_lm/models/zaya.py): ModelArgs + 12 module classes mirroring ZayaForCausalLM
- All 2,483 HF safetensors load via `mlx_lm.load("Zyphra/ZAYA1-8B")` with strict=True
- Param count: **8,840,489,464** (exact match with HF total bf16 / 2)
- 8/8 weight-loading tests pass
- mlx-lm fork pushed to https://github.com/zappleg8/mlx-lm on branch `zaya1`

## Headline finding from Phase 0

**The architecture is not a Mamba+Attention hybrid.** What was thought to be SSM is **CCA** (Compressed Causal Attention) — a custom attention variant with a depthwise 1D causal conv on Q+K and a time-shifted V stream. R1 (custom SSM parity unreachable) is eliminated.

## Phase 1 wrinkles found and resolved

- MLX `nn.Sequential` exposes children as `.layers.0.weight`; HF safetensors store `.0.weight`. Resolved in `sanitize` via regex insertion of `.layers` into `conv_qk` and `router_mlp` paths.
- PyTorch Conv1d weight layout `(out, in/groups, kernel)` differs from MLX `(out, kernel, in/groups)`. Resolved in `sanitize` via `v.transpose(0, 2, 1)` on `conv_qk` weights.
- mlx-lm uses `importlib.import_module(f"mlx_lm.models.{model_type}")` for dispatch. No registry edit needed; just create `zaya.py` in the right place.

## What's next

**Phase 2: partial RoPE wrapper.** Plan to be written. Key tasks:

1. Implement `partial_rope` helper that applies `mlx-lm`'s RoPE primitive to the first `head_dim × partial_rotary_factor = 64` features of Q/K, passing the remaining 64 features through unchanged.
2. Test: load reference activations for `self_attn_qkv_q_out` (post-RoPE Q) and verify per-tensor parity vs MLX implementation, max abs diff < 1e-3 cosine sim > 0.999.

The MLX skeleton's stub `__call__` will need to be partially implemented for Phase 2 — at minimum, embedding + first ATT layer's Q/K projection + partial RoPE.

## Blockers

None.

## Reference activation paths

- Index: [`reference/MANIFEST.md`](reference/MANIFEST.md)
- Architecture catalog + shape inventory: [`reference/notes/zaya-architecture.md`](reference/notes/zaya-architecture.md)
- HF weight key inventory: [`reference/notes/hf-weight-keys.md`](reference/notes/hf-weight-keys.md)
- mlx-lm conventions: [`reference/notes/mlx-lm-conventions.md`](reference/notes/mlx-lm-conventions.md)
- Install log: [`reference/notes/install-log.md`](reference/notes/install-log.md)
- Dumps: `reference/activations/{smoke,reasoning_short,long_context_seed}/` (gitignored)

## Specs and plans

- Design (R1): [`docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md`](docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md)
- Phase 0 plan: [`docs/superpowers/plans/2026-05-06-phase0-reference-scaffolding.md`](docs/superpowers/plans/2026-05-06-phase0-reference-scaffolding.md)
- Phase 1 plan: [`docs/superpowers/plans/2026-05-06-phase1-skeleton-and-weight-loading.md`](docs/superpowers/plans/2026-05-06-phase1-skeleton-and-weight-loading.md)
- Phase 2+ plans: not yet written. Plan 3 will follow Phase 1 sign-off.

## Repos

- **zaya1-mlx** (this repo): https://github.com/zappleg8/zaya1-mlx (public, main branch)
- **mlx-lm fork**: https://github.com/zappleg8/mlx-lm (zaya1 branch, ready for upstream PR)

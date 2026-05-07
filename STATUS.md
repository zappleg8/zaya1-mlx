# Status

**Last updated:** 2026-05-06

## Current phase

**Phase 2 — COMPLETE.** Phase 3 (CCA forward implementation) not yet started.

## What's done

Phase 0 (reference scaffolding):
- Reference uv venv at `reference/.venv` with `torch==2.5.1` (CPU) + `transformers==4.57.1` from `Zyphra/transformers @ zaya1` (commit `f0ab5bef`)
- ZAYA1-8B weights downloaded (16 GB)
- Source code read end-to-end; architecture cataloged at [`reference/notes/zaya-architecture.md`](reference/notes/zaya-architecture.md)
- All 5 open architectural questions from spec §5 resolved
- Spec amended (R1): no SSM in ZAYA1; CCA replaces it; layer schedule is 1:1 ATT/MoE alternation
- `dump_activations.py` + 5/5 tests passing
- 3 reference dumps captured (smoke, reasoning_short, long_context_seed)

Phase 1 (skeleton + weight loading):
- HF weight key inventory at [`reference/notes/hf-weight-keys.md`](reference/notes/hf-weight-keys.md): 28 unique key patterns, 2,483 total tensors
- mlx-lm conventions at [`reference/notes/mlx-lm-conventions.md`](reference/notes/mlx-lm-conventions.md)
- Validation venv at `validation/.venv` with MLX 0.31.2 + editable mlx-lm
- Skeleton [`~/code/personal/mlx-lm/mlx_lm/models/zaya.py`](../mlx-lm/mlx_lm/models/zaya.py): ModelArgs + 12 module classes
- All 2,483 HF safetensors load via `mlx_lm.load("Zyphra/ZAYA1-8B")` with strict=True
- Param count: **8,840,489,464** (exact match with HF total bf16 / 2)
- 8/8 weight-loading tests pass

Phase 2 (partial RoPE):
- `dump_activations.py` augmented to capture 2-tuple outputs as `_0`/`_1` (rotary cos and sin both saved)
- Reference dumps refreshed (3,087 tensors per prompt, +41 vs Phase 0)
- 3/3 partial RoPE parity tests pass: synthetic input, dumped cos/sin reproducibility, dumped Q with reference cos/sin
- Confirmed: mlx-lm's built-in `nn.RoPE(dims=64, base=5e6, traditional=False)` correctly implements Zaya's partial RoPE within bf16 rounding noise — no custom helper needed
- `nn.RoPE` added to `ZayaAttention` skeleton (Phase 4 will use it)

## Headline finding from Phase 0

**The architecture is not a Mamba+Attention hybrid.** What was thought to be SSM is **CCA** (Compressed Causal Attention) — a custom attention variant with a depthwise 1D causal conv on Q+K and a time-shifted V stream. R1 (custom SSM parity unreachable) is eliminated.

## Phase 2 finding

PyTorch stores cos/sin as bf16 in the model. MLX computes them in fp32 internally — more precise. This creates a small bit-mismatch (~2e-3 on cos, ~5e-3 on post-RoPE Q) that will appear in all later parity tests. Documented as `BF16_COS_SIN_TOL` and `BF16_POST_ROPE_TOL` constants. This is the noise floor for all subsequent parity tests against reference dumps.

## What's next

**Phase 3: CCA forward implementation.** The most novel piece of the model. Plan to be written. Key components:

1. Two-stage depthwise 1D causal conv on concatenated Q+K (kernel sizes `cca_time0=2`, `cca_time1=2`)
2. Mean residual mixing (pre-conv Q/K averaged with post-conv)
3. Two-stream V (current hidden state + time-shifted)
4. Per-head L2 normalization with learnable temperature
5. Output (Q, K, V) ready for standard attention compute

Gate: parity on `self_attn_qkv_q`, `_k`, `_v` reference tensors at L0, L40, L78 (within bf16 noise).

## Blockers

None.

## Reference activation paths

- Index: [`reference/MANIFEST.md`](reference/MANIFEST.md)
- Architecture catalog + shape inventory: [`reference/notes/zaya-architecture.md`](reference/notes/zaya-architecture.md)
- HF weight key inventory: [`reference/notes/hf-weight-keys.md`](reference/notes/hf-weight-keys.md)
- mlx-lm conventions: [`reference/notes/mlx-lm-conventions.md`](reference/notes/mlx-lm-conventions.md)
- Install log: [`reference/notes/install-log.md`](reference/notes/install-log.md)
- Dumps: `reference/activations/{smoke,reasoning_short,long_context_seed}/` (gitignored, 3,087 tensors per prompt)

## Specs and plans

- Design (R1): [`docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md`](docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md)
- Phase 0 plan: [`docs/superpowers/plans/2026-05-06-phase0-reference-scaffolding.md`](docs/superpowers/plans/2026-05-06-phase0-reference-scaffolding.md)
- Phase 1 plan: [`docs/superpowers/plans/2026-05-06-phase1-skeleton-and-weight-loading.md`](docs/superpowers/plans/2026-05-06-phase1-skeleton-and-weight-loading.md)
- Phase 2 plan: [`docs/superpowers/plans/2026-05-06-phase2-partial-rope.md`](docs/superpowers/plans/2026-05-06-phase2-partial-rope.md)
- Phase 3+ plans: not yet written.

## Repos

- **zaya1-mlx**: https://github.com/zappleg8/zaya1-mlx (public, main branch)
- **mlx-lm fork**: https://github.com/zappleg8/mlx-lm (zaya1 branch)

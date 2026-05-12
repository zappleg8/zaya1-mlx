# Status

**Last updated:** 2026-05-06

## Current phase

**Phase 5 — COMPLETE.** Phase 6 (MoE forward — `ZayaRouter` + `MLP` + `SequentialMLP` + `ZayaBlock` with skip-expert MoD) not yet started.

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

Phase 5 (ATT decoder layer):
- `ZayaDecoderATTLayer.__call__` composes ResidualScaling + residual init/merge + input_norm (with dtype cast to/from fp32) + ZayaAttention. Returns `(hidden_states,), residual, prev_router_hidden_states`.
- 4/4 layer tests pass: end-to-end L0 hidden_states output vs `L0_self_attn_out`; L0 residual output vs `L0_layer_out` (which the dump captures as the residual, not hidden_states, due to the layer returning a 3-tuple); residual fp32 dtype check; non-first-layer synthetic residual+merge path.
- Validation suite at 21/21.
- Discovered the dump's `L{i}_layer_out` semantics: it's the residual stream (fp32), not the layer's hidden_states output. Documented in the test.

Phase 4 (ZayaAttention forward):
- `ZayaAttention.__call__` composes CCA(qkv) → reshape to (B, n_heads, S, D) → partial RoPE → mlx-lm `scaled_dot_product_attention` (handles GQA automatically) → o_proj
- 3/3 attention forward parity tests pass at L0, L40, L78 within bf16 noise floor (max diffs: 0.34, 0.09, 0.16 vs tolerance 0.5)
- Validation suite at 17/17

Infrastructure fix:
- Discovered & fixed an OOM footgun: each validation test file had its own `@pytest.fixture(scope="session")` for `loaded_model`. pytest treats same-name fixtures in different files as independent — so the full suite loaded the 17 GB model 3-4 times. Consolidated to `validation/conftest.py` so the model loads exactly once per invocation. Full suite now runs in 6.3 s with no memory issues.

HF config update handling:
- Zyphra updated config.json on 2026-05-11: `num_attention_heads` 16→8 (now reflects effective Q heads), `cca_num_q_heads` removed, explicit `head_dim` added. Weights bit-identical. ModelArgs updated to handle both old and new schemas (uses `__post_init__` for resolution).

Phase 3 (CCA forward):
- `CCA.__call__` implemented in HF `(B, S, H)` layout: linear_q/k projections, pre-conv mean residual, two-stage depthwise/grouped Conv1d on packed [Q, K], two-stream V (current + time-shifted), per-head L2-normalized Q/K with learnable temperature
- 3/3 CCA forward parity tests pass at L0, L40, L78 within bf16 noise floor
- bf16 noise budget for CCA documented: Q ~5e-2, K ~2.5e-1 (K head 0 has large pre-norm magnitudes ~30k where bf16 ULP is 128). V is exact (linear projections only, no normalization).
- Discovered and fixed: dump tests were using prompt `smoke` and clobbering the full reference dump. Switched to dedicated `_test` prompt.
- Validation suite at 14/14 (8 weight loading + 3 partial RoPE + 3 CCA forward)

## Headline finding from Phase 0

**The architecture is not a Mamba+Attention hybrid.** What was thought to be SSM is **CCA** (Compressed Causal Attention) — a custom attention variant with a depthwise 1D causal conv on Q+K and a time-shifted V stream. R1 (custom SSM parity unreachable) is eliminated.

## Phase 2 finding

PyTorch stores cos/sin as bf16 in the model. MLX computes them in fp32 internally — more precise. This creates a small bit-mismatch (~2e-3 on cos, ~5e-3 on post-RoPE Q) that will appear in all later parity tests. Documented as `BF16_COS_SIN_TOL` and `BF16_POST_ROPE_TOL` constants. This is the noise floor for all subsequent parity tests against reference dumps.

## What's next

**Phase 6: MoE forward** (`ZayaRouter` with EDA + 16 experts via `SequentialMLP` + `ZayaBlock` with MoD skip-expert routing). The router has been the most architecturally interesting piece — top-1 expert selection with load-balancing biases and a learnable "skip" 17th option that lets tokens bypass the MLP entirely.

Gate: parity on `L1_zaya_block_router_*` and `L1_zaya_block_out` reference tensors at the first MoE layer (L1, no EDA) and a later MoE layer (L3, with EDA).

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
- Phase 3 plan: [`docs/superpowers/plans/2026-05-06-phase3-cca-forward.md`](docs/superpowers/plans/2026-05-06-phase3-cca-forward.md)
- Phase 4 plan: [`docs/superpowers/plans/2026-05-06-phase4-zaya-attention-forward.md`](docs/superpowers/plans/2026-05-06-phase4-zaya-attention-forward.md)
- Phase 5 plan: [`docs/superpowers/plans/2026-05-06-phase5-att-decoder-layer.md`](docs/superpowers/plans/2026-05-06-phase5-att-decoder-layer.md)
- Phase 6+ plans: not yet written.

## Repos

- **zaya1-mlx**: https://github.com/zappleg8/zaya1-mlx (public, main branch)
- **mlx-lm fork**: https://github.com/zappleg8/mlx-lm (zaya1 branch)

# Status

**Last updated:** 2026-05-06

## Current phase

**Phase 0 — COMPLETE.** Phase 1 (skeleton + weight loading in mlx-lm fork) not yet started.

## What's done

Phase 0 (reference scaffolding):
- Reference uv venv set up at `reference/.venv` with `torch==2.5.1` (CPU) + `transformers==4.57.1` from `Zyphra/transformers @ zaya1` (commit `f0ab5bef`)
- ZAYA1-8B weights downloaded (`~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B/`, 16 GB)
- Source code read end-to-end (`modular_zaya.py` 2,316 LOC + `configuration_zaya.py` 126 LOC)
- Architecture cataloged at [`reference/notes/zaya-architecture.md`](reference/notes/zaya-architecture.md)
- All 5 open architectural questions from spec §5 resolved
- Spec amended (R1): no SSM in ZAYA1; CCA replaces it; layer schedule is 1:1 ATT/MoE alternation
- `reference/dump_activations.py` implemented + 5 pytest tests passing
- 3 reference dumps captured with 3,046 tensors each:
  - `smoke` (7 input tokens) — 57 MB
  - `reasoning_short` (22 tokens) — 158 MB
  - `long_context_seed` (78 tokens) — 531 MB
- `reference/MANIFEST.md` indexes available dumps
- Repo public at https://github.com/zappleg8/zaya1-mlx

## Headline finding from Phase 0

**The architecture is not a Mamba+Attention hybrid.** What was thought to be SSM is **CCA** (Compressed Causal Attention) — a custom attention variant with a depthwise 1D causal conv on Q+K and a time-shifted V stream. R1 (custom SSM parity unreachable, originally the highest risk) is eliminated. The new highest-uncertainty piece is CCA itself (R1' in the amended risk register).

## What's next

**Phase 1: skeleton + weight loading in mlx-lm fork.** Plan to be written. Key tasks:

1. Define `ModelArgs` dataclass mirroring `ZayaConfig` (in `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`).
2. Stub the full module hierarchy with correct shapes but no forward logic.
3. Write `sanitize(weights)` that maps HF safetensors keys → MLX nn names.
4. Verify all 4 safetensors shards load with strict=True; every weight finds a home; no leftovers.
5. Special-case `tie_word_embeddings`: don't load `lm_head.weight` separately; alias from `embed_tokens.weight`.

Gate for Phase 1: `weights = mx.load(...); model.load_weights(weights, strict=True)` succeeds.

## Blockers

None.

## Reference activation paths

- Index: [`reference/MANIFEST.md`](reference/MANIFEST.md)
- Architecture catalog + shape inventory: [`reference/notes/zaya-architecture.md`](reference/notes/zaya-architecture.md)
- Install log: [`reference/notes/install-log.md`](reference/notes/install-log.md)
- Dumps: `reference/activations/{smoke,reasoning_short,long_context_seed}/` (gitignored)

## Specs and plans

- Design (R1): [`docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md`](docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md)
- Phase 0 plan: [`docs/superpowers/plans/2026-05-06-phase0-reference-scaffolding.md`](docs/superpowers/plans/2026-05-06-phase0-reference-scaffolding.md)
- Phase 1+ plans: not yet written. Plan 2 will follow Phase 0 sign-off.

## mlx-lm fork

- Path: `~/code/personal/mlx-lm`
- Origin: `https://github.com/zappleg8/mlx-lm`
- Active branch: `zaya1` (clean; no model code yet)
- Bootstrap notes: [`mlx-lm/CLAUDE.md`](../mlx-lm/CLAUDE.md)

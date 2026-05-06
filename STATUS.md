# Status

**Last updated:** 2026-05-06

## Current phase

Phase 0 — Reference scaffolding (not yet started).

## What's done

- Design doc written and approved: [`docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md`](docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md)
- Repo initialized
- `ml-explore/mlx-lm` forked locally (see CLAUDE.md for path)

## What's next

Implementation plan generation → Phase 0 reference scaffolding (uv venv, install Zyphra transformers fork, weights download, `dump_activations.py`).

## Blockers

None.

## Open architectural facts to confirm during implementation

These are facts to extract from `Zyphra/transformers @ zaya1` source code, not design choices:

- Layer schedule — are SSM and attention layers interleaved (Jamba-style), or is every layer hybrid?
- MoE coverage — every layer or a subset?
- `ZayaRMSNorm` semantics — exact difference from stock RMSNorm, if any.
- `scale_residual_merge` formula.
- EDA exact form (depth-averaging window, weighting).

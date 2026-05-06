# ZAYA1-8B → MLX Port: Design

**Date:** 2026-05-06
**Status:** Approved (brainstorm), pending implementation plan

---

## 1. Goal

Produce a faithful Apple MLX port of Zyphra's `ZAYA1-8B` (released 2026-05-06) and publish it on Hugging Face as `mlx-community/ZAYA1-8B` (BF16) and `mlx-community/ZAYA1-8B-4bit`. The port must be numerically equivalent to the PyTorch reference at every layer boundary (within tolerance) before we ship.

This is the first MLX port of the model. The work serves a strategic objective (visibility with Zyphra as a contributor) and a personal objective (first-principles understanding of MoE + MoD + EDA + custom SSM hybrid architectures, sufficient to defend the implementation in technical conversation).

## 2. Non-Goals (this spec)

- Tool-call XML parser and agent harness work (separate spec; depends on this one)
- BFCL / τ-bench benchmarks (separate spec)
- GGUF or llama.cpp port (different framework, different rate of work)
- Training, fine-tuning, or LoRA support
- Multi-GPU / distributed inference

## 3. Strategic Context

ZAYA1-8B is novel along several axes (custom SSM, MoE with top-1 routing, Mixture-of-Depths skip routing, Exponential Depth Averaging, partial RoPE, scale_residual_merge). No MLX port exists, no quantized version exists, and no inference providers are listed. Being first with a working `mlx-community/ZAYA1-8B-4bit` plus a credible upstream PR to `ml-explore/mlx-lm` is a clear, durable signal.

The MLX port is also the prerequisite for local agentic experiments on Apple Silicon. The model card explicitly markets test-time-compute harnesses, and ZAYA1's relative weakness vs. Qwen3.5-4B is on agentic benchmarks (BFCL-v4 39.2, τ² 43.1) — there is real room to demonstrate harness engineering on top of a local checkpoint.

## 4. Hardware Target

- MacBook Pro M3 Max, 36 GB unified memory
- BF16 weights ≈ 17 GB; 4-bit weights ≈ 4.5 GB
- 760 M active parameters per forward pass (top-1 MoE routing) → fast despite 8.4 B total
- Reference PyTorch and MLX **never resident in the same Python process** (RAM constraint)

## 5. Architecture Overview

Verified from `Zyphra/transformers @ zaya1` and `Zyphra/ZAYA1-8B/config.json`. Facts I have committed to:

- `hidden_size`: 2048
- `num_hidden_layers`: 80
- `num_attention_heads`: 16, `num_key_value_heads`: 2 (GQA, 8:1)
- `kv_channels`: 128
- `ffn_hidden_size`: 4096
- `vocab_size`: 262272
- `max_position_embeddings`: 131072
- `num_experts`: 16, `moe_router_topk`: 1
- `partial_rotary_factor`: 0.5
- `rope_theta`: 5000000, `rope_scaling`: false
- `zaya_use_mod`: true (Mixture of Depths — skip-expert)
- `zaya_use_eda`: true (Exponential Depth Averaging)
- `zaya_mlp_expansion`: 256
- `mamba_cache_dtype`: float32
- `residual_in_fp32`: true
- `scale_residual_merge`: true
- `normalization`: RMSNorm
- `activation_func`: swiglu

Six novel components require ports, in dependency order:

1. **`ZayaRMSNorm`** — verify whether structurally identical to standard RMSNorm or has differing semantics.
2. **Partial RoPE** — apply `mlx.core.fast.rope` to the first 50% of head dim, concat the unrotated half. mlx-lm has no built-in for this.
3. **`ZayaAttention`** — GQA (16 Q / 2 KV), partial RoPE, integrates with mlx-lm's KV cache classes.
4. **`ZayaSSM`** — custom Mamba-variant recurrent scan. State kept in fp32 even when activations are bf16. Highest implementation risk.
5. **MoE stack** — `ZayaRouter` (linear + softmax + top-1) feeding `ZayaMoE` (16 experts + optional skip expert when MoD active).
6. **EDA path** — `ZayaRMSNorm`-normalized depth-averaged hidden state, fires from the second eligible layer onward.

Plus `scale_residual_merge` (a residual scaling factor applied at merge — exact formula to be read from PyTorch source).

**Open architectural facts to confirm during implementation:**

- Layer schedule — are SSM and attention layers interleaved (Jamba-style), or every layer hybrid? Read from `modeling_zaya.py` config or layer-construction logic.
- MoE coverage — every layer or a subset?
- `ZayaRMSNorm` semantics — exact difference from stock RMSNorm, if any.
- `scale_residual_merge` formula.
- EDA exact form (depth-averaging window, weighting).

These are facts to extract from source code, not design choices.

## 6. Repo Structure

Two repositories, distinct purposes.

### 6.1 `~/code/personal/mlx-lm` (fork of `ml-explore/mlx-lm`)

Model code only. Designed to merge cleanly upstream as a PR.

```
mlx-lm/
  mlx_lm/
    models/
      zaya.py            # all classes for the port
      __init__.py        # registers "zaya" → zaya.Model
  CLAUDE.md              # one-page pointer to zaya1-mlx
```

`zaya.py` contents, in dependency order:

- `ModelArgs` dataclass mirroring `configuration_zaya.py`
- `ZayaRMSNorm`
- `partial_rope(...)` helper
- `ZayaAttention` (GQA, partial RoPE, KV cache)
- `ZayaSSM`
- `ZayaRouter`
- `ZayaMoE` (16 experts + optional skip expert + EDA path)
- `ZayaDecoderLayer`
- `ZayaModel` (embedding, 80 layers, final norm)
- `Model` (mlx-lm canonical name; adds lm_head)
- `sanitize(weights)` (HF → MLX weight key remapping)

### 6.2 `~/code/personal/zaya1-mlx` (new repo)

Everything around the model: validation harness, HF upload, future agent harness.

```
zaya1-mlx/
  CLAUDE.md                # session bootstrap; first read for any new session
  STATUS.md                # current phase, what's done, what's next, what's blocked
  docs/superpowers/specs/  # this design doc + implementation plan
  reference/
    setup.sh               # uv venv + Zyphra transformers fork install
    dump_activations.py    # PyTorch forward + hook every submodule, write .npy
    activations/           # gitignored — large
    MANIFEST.md            # which prompts × layers we have refs for
  validation/
    compare.py             # MLX out vs .npy ref: max abs diff, cosine similarity
    test_layer_parity.py   # pytest-driven per-layer parity tests
  scripts/
    convert_4bit.sh
    upload_hf.sh
  zaya1_mlx/               # thin Python package re-exporting the model
    __init__.py
  pyproject.toml
```

### 6.3 Cross-session continuity

The split must survive context window resets. Mechanisms:

- `zaya1-mlx/CLAUDE.md` — single source of truth. Points to the fork, the design doc, the status doc, the activations manifest, the harness entry points.
- `mlx-lm/CLAUDE.md` — small. Says "this is a fork; for context see `~/code/personal/zaya1-mlx`".
- `zaya1-mlx/STATUS.md` — updated at the end of every work session: current phase, gate status, blockers.
- Design doc and implementation plan live in `zaya1-mlx/docs/superpowers/specs/`, versioned with the work.

## 7. Validation Strategy

Reference-driven, layer-by-layer, asynchronous between frameworks.

### 7.1 Phase A — PyTorch reference dumps (offline)

In a dedicated `uv` venv inside `zaya1-mlx/reference/`:

1. Install `Zyphra/transformers @ zaya1` (CPU torch acceptable).
2. Load ZAYA1-8B (BF16) from local HF cache.
3. Run a fixed 32-token prompt forward pass.
4. Forward hooks on every major submodule of every layer record outputs to `reference/activations/<prompt_hash>/L{i}_{module}.npy`.
5. Modules captured (per layer): input norm, attn Q/K/V projections, attn output, SSM input projection, SSM dt, SSM state, SSM output, MoE router logits, MoE per-expert outputs, MoE merged output, EDA input, EDA output, residual merge output.
6. `MANIFEST.md` records prompt text, hash, capture date, torch version.

PyTorch is loaded once per design checkpoint, dumps are reused indefinitely.

### 7.2 Phase B — MLX comparison (iterative)

Per submodule, after implementing in MLX:

1. Build the corresponding submodule in isolation, load the same weights.
2. Feed it the saved input tensor from `.npy`.
3. Compare output to saved reference.
4. Tolerance buckets:
   - Tensors computed in fp32 in PyTorch (e.g. SSM state per `mamba_cache_dtype: float32`, residuals if `residual_in_fp32: true`): max abs diff < 1e-4, cosine similarity > 0.9999.
   - Tensors computed in bf16 (most attention and MoE intermediates): max abs diff < 1e-3, cosine similarity > 0.999.
   - Final logits: top-5 token ranking must match exactly; greedy argmax must match.

Failures stop forward progress until resolved.

### 7.3 RAM discipline

Never load PyTorch and MLX in the same process. Reference dumps are the sole interface between them. This is non-negotiable on 36 GB.

### 7.4 PyTorch reference fallback

If `Zyphra/transformers @ zaya1` does not build cleanly on macOS (likely failure modes: triton dependency, CUDA-only kernels gated behind `torch.cuda.is_available()` checks), reference dumps run on a one-off Linux GPU instance (Modal or Vast.ai for a few hours). Dumps are write-once-read-many, so this is acceptable.

## 8. Implementation Phases

Each phase has a single correctness gate. We do not advance until the gate passes.

| Phase | Work | Gate |
|---|---|---|
| 0 | Reference scaffolding: venv, weights download, `dump_activations.py` | PyTorch produces output, `.npy` files on disk, manifest written |
| 1 | `ModelArgs` + empty `Model` shell + `sanitize(weights)` | All 4 safetensors shards load, every weight finds a home, no leftovers |
| 2 | `ZayaRMSNorm` + `partial_rope` | Per-tensor parity vs reference, < 1e-4 max abs diff |
| 3 | `ZayaAttention` end-to-end + KV cache | Parity at L0, L40, L79 |
| 4 | `ZayaSSM` (custom recurrent scan) | Parity at L0 (or earliest SSM layer), L40, L79 |
| 5 | `ZayaRouter` + `ZayaMoE` (with skip expert + EDA) | Parity at L1 (first EDA-eligible), L40, L79 |
| 6 | Full forward: 80 stacked layers + embedding + lm_head | Top-5 logit ranking matches PyTorch; greedy next-token matches |
| 7 | `mlx_lm.generate` integration | Coherent output on model-card example prompts; tokens/sec measured |
| 8 | 4-bit conversion via `mlx_lm.convert` | Perplexity within 3% of BF16 on ~1000 tokens of held-out text |
| 9 | HF upload + Zyphra issue/PR | `mlx_lm.load("mlx-community/ZAYA1-8B-4bit")` works from a fresh shell |

Each phase commits to its own feature branch. Bugs found in earlier phases force re-validation of those phases.

**Stop conditions:**

- Phase 4 cannot reach SSM parity within ~3 dedicated work sessions (a session ≈ 2–3 focused hours) → stop, escalate (Zyphra issue/discord), do not push wrong-output gibberish forward.
- Phase 8 4-bit perplexity regresses beyond 3% → keep router and EDA in higher precision, retry; if still failing, ship BF16 only and document the quant gap.

## 9. Risks

| ID | Risk | Likelihood × Impact | Mitigation |
|---|---|---|---|
| R1 | Custom SSM cannot reach parity | High × High | Dump SSM internals separately (input proj, dt, state, output proj), not just layer output, to localize bugs. Hard stop at 3 sessions. |
| R2 | Zyphra transformers fork doesn't build on macOS | Medium × High | Try CPU-only torch first; fall back to one-off Linux GPU instance for reference dumps |
| R3 | RAM pressure under simultaneous PyTorch + MLX | Medium × Medium | Already mitigated by offline-dump workflow |
| R4 | `mlx_lm.convert` mishandles MoE/SSM weights | Medium × Medium | Study `mlx_lm/models/jamba.py` (similar hybrid). Add custom quant-skip predicate for router and SSM A/B params if needed |
| R5 | HF safetensors → MLX weight key mismatches | Low × Medium | `sanitize` logs every unmapped key; never silently drop weights |
| R6 | `mlx-community` HF org membership unknown | Low × Low | Check at start of Phase 9; fall back to personal namespace, request access in parallel |
| R7 | 4-bit quant disproportionately damages MoD/EDA | Low × Medium (unknown) | Phase 8 gate already covers; mitigation built into stop conditions |
| R8 | Zyphra doesn't notice the work | Low × High (strategic) | Issue + PR linking back, HF model card credits Zyphra, mlx-community visibility, optional follow-up writeup |

## 10. Success Criteria

This spec is complete when all of:

1. `mlx_lm.generate --model mlx-community/ZAYA1-8B-4bit --prompt "..."` produces coherent output on a fresh M3 Max, with no manual setup beyond `pip install mlx-lm`.
2. The MLX 4-bit checkpoint passes the perplexity gate (≤ 3% delta vs BF16 on held-out text).
3. An issue or PR exists on `Zyphra/transformers` linking to the MLX checkpoints and the upstream PR to `ml-explore/mlx-lm`.
4. The implementation is faithful enough that the author can answer first-principles questions about every novel component (EDA, MoD skip routing, partial RoPE, scale_residual_merge, custom SSM scan) without reference to the original code.

## 11. Out-of-Scope Future Work

Tracked for the next spec cycle, not this one:

- `zaya_xml` tool-call parser port (PyTorch source: vLLM fork)
- Local agent harness with tool calling
- BFCL-v4 and τ-bench evaluation runs
- Optional follow-up: GGUF / llama.cpp port

## 12. Open Questions Resolved During Brainstorm

| Question | Decision |
|---|---|
| Code organization | Hybrid: fork mlx-lm for the model code, separate `zaya1-mlx` repo for everything else |
| Validation strategy | Reference-driven (layer-by-layer parity vs PyTorch) |
| v1 scope | Strategic — implementation + 4-bit quant + HF upload + Zyphra notification |
| SSM stop condition | Yes, hard stop if parity unreachable |
| Quant tolerance | 3% perplexity delta on held-out text |

# ZAYA1-8B → MLX Port: Design

**Date:** 2026-05-06
**Status:** Approved (brainstorm), implementation in progress
**Revision:** R1 (2026-05-06) — §5, §7, §8, §9 amended after reading the PyTorch source. Headline correction: there is no SSM in ZAYA1; what was thought to be Mamba is **CCA** (Compressed Causal Attention). See `reference/notes/zaya-architecture.md` for the source-grounded architecture catalog that supersedes any earlier claims.

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

## 5. Architecture Overview (R1 — corrected after source read)

All facts in this section are now grounded in `Zyphra/transformers @ zaya1` (commit `f0ab5bef`) and verified against the live `Zyphra/ZAYA1-8B/config.json`. Detailed catalog at `reference/notes/zaya-architecture.md`.

**Config facts (unchanged from R0):**

- `hidden_size`: 2048
- `num_hidden_layers`: 80
- `num_attention_heads`: 16, `num_key_value_heads`: 2
- `cca_num_q_heads`: 8 (effective Q heads after CCA compression)
- `kv_channels`: 128 (head_dim)
- `ffn_hidden_size`: 4096 (with gated linear unit, effective output is 2048)
- `vocab_size`: 262272
- `max_position_embeddings`: 131072
- `num_experts`: 16, `moe_router_topk`: 1, `zaya_mlp_expansion`: 256 (router internal dim)
- `partial_rotary_factor`: 0.5 (rotary_dim = 64 of 128 head dims)
- `rope_theta`: 5000000, `rope_scaling`: false
- `zaya_use_mod`: true (skip-expert), `zaya_use_eda`: true
- `cca_time0`: 2, `cca_time1`: 2 (CCA conv kernel sizes)
- `mamba_cache_dtype`: float32 (misnamed — applies to CCA conv state, not an SSM)
- `residual_in_fp32`: true, `scale_residual_merge`: true
- `tie_word_embeddings`: true (lm_head weight = embed_tokens weight)
- `normalization`: RMSNorm, `activation_func`: swiglu

**Layer schedule:** strictly 1:1 alternating, starting with attention. Layer 0=ATT, 1=MoE, 2=ATT, …, 78=ATT, 79=MoE. So 40 ATT layers + 40 MoE layers. `modular_zaya.py:1663-1681` is the source of truth.

**Components requiring port (dependency order):**

1. **Partial RoPE** — mlx-lm has no built-in. Apply RoPE to the first 64 of 128 head dims, pass the rest through unchanged.
2. **CCA (Compressed Causal Attention)** — the most novel piece. For even layers. Two-stage depthwise 1D causal conv (kernel 2 each) on concatenated [Q, K] along sequence; per-head L2 norm with learnable per-KV-head temperature; two-stream V (current hidden state + one-step-back hidden state). Compresses to 8 effective Q heads.
3. **ZayaAttention** — wraps CCA + standard scaled dot product attention with GQA (8 Q heads, 2 KV heads, group_size=4) and partial RoPE.
4. **ResidualScaling** — per-feature affine `(stream + bias) * scale` on both residual and hidden_states streams before merging. First layer skips the residual transform.
5. **ZayaDecoderATTLayer** — composes ResidualScaling + residual merge + RMSNorm + ZayaAttention, with the residual stream threaded through (separate from hidden_states).
6. **ZayaRouter** — for odd layers. Linear down-projection (2048→256) + optional EDA gate (`hs += prev_router_states * learnable_scale`, gated off for first MoE layer) + RMSNorm + 3-layer GELU MLP (D→D→D→17) + softmax + top-1 selection biased by `balancing_biases` (load balancing).
7. **MLP / SequentialMLP** — single SwiGLU expert; container of 16 experts.
8. **ZayaBlock** — sorts tokens by chosen expert, runs experts, MoD-skip-expert branch (skip = pass through unchanged), un-permutes, gates by route_prob.
9. **ZayaDecoderMLPLayer** — composes ResidualScaling + residual merge + RMSNorm + ZayaBlock.
10. **ZayaModel** — embedding + 80 alternating decoder layers + final ResidualScaling + final RMSNorm.
11. **ZayaForCausalLM** — adds lm_head with weight tied to embed_tokens.

Components that **do not** require custom code (use mlx-lm built-ins):

- **RMSNorm** — `ZayaRMSNorm` is identical to standard T5/Llama RMSNorm. Drop-in `nn.RMSNorm`.
- **RoPE base** — `ZayaRotaryEmbedding` is identical to `Glm4RotaryEmbedding` (standard RoPE). The "partial" part is a slice/concat wrapper around mlx-lm `nn.RoPE`.
- **Causal mask** — vanilla copies from Phi3/Mistral. mlx-lm's standard mask creation suffices.
- **SwiGLU** — standard.

**Components NOT in the architecture** (corrections to R0 spec):

- ❌ No SSM, no Mamba scan, no recurrent state-space layers.
- ❌ EDA is not on main hidden states. EDA is on the router's 256-dim hidden states, threaded through MoE layers only.
- ❌ EDA is not exponential averaging. It is a learnable per-feature affine combination.

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

## 8. Implementation Phases (R1 — restructured after source read)

Each phase has a single correctness gate. We do not advance until the gate passes.

| Phase | Work | Gate |
|---|---|---|
| 0 | Reference scaffolding: venv, weights download, source read, `dump_activations.py` | PyTorch produces output, `.npy` files on disk, manifest + architecture doc written |
| 1 | `ModelArgs` + empty `Model` shell in mlx-lm fork + `sanitize(weights)` | All 4 safetensors shards load, every weight finds a home, no leftovers; `tie_word_embeddings` handled |
| 2 | Partial RoPE wrapper around mlx-lm's `nn.RoPE` | Per-tensor parity vs reference Q/K-after-RoPE, < 1e-4 max abs diff |
| 3 | **CCA** (depthwise conv + time-shift V₂ + L2 norm + per-head temp) | Parity on `qkv_q_out`, `qkv_k_out`, `qkv_v_out` at L0, L40, L78 (all even layers) |
| 4 | `ZayaAttention` (CCA + standard SDPA + GQA + partial RoPE + KV cache) | Parity on `self_attn_out` at L0, L40, L78 |
| 5 | `ResidualScaling` + `ZayaDecoderATTLayer` | Parity on full ATT layer output at L0, L40, L78 |
| 6 | `ZayaRouter` (with EDA + balancing biases) + `MLP` + `SequentialMLP` + `ZayaBlock` (with MoD skip-expert) | Parity on `router_out`, `zaya_block_out` at L1, L41, L79 |
| 7 | `ZayaDecoderMLPLayer` | Parity on full MoE layer output at L1, L41, L79 |
| 8 | `ZayaModel` forward (80-layer alternation + residual+router_hs threading + final norm) | Parity on `final_norm_out` |
| 9 | `ZayaForCausalLM` (lm_head, tied weights) + end-to-end logits | Top-5 logit ranking matches PyTorch; greedy next-token matches |
| 10 | `mlx_lm.generate` integration | Coherent output on model-card example prompts; tokens/sec measured on M3 Max |
| 11 | 4-bit conversion via `mlx_lm.convert` | Perplexity within 3% of BF16 on ~1000 tokens of held-out text |
| 12 | HF upload + Zyphra issue/PR | `mlx_lm.load("mlx-community/ZAYA1-8B-4bit")` works from a fresh shell |

Each phase commits to its own feature branch. Bugs found in earlier phases force re-validation of those phases.

**Stop conditions:**

- Phase 3 cannot reach CCA parity within ~3 dedicated work sessions (a session ≈ 2–3 focused hours) → stop, escalate. CCA replaces SSM as the highest-uncertainty piece because of its custom depthwise-grouped conv pattern and per-head L2 normalization.
- Phase 11 4-bit perplexity regresses beyond 3% → keep router and CCA temperature/conv weights in higher precision, retry; if still failing, ship BF16 only and document the quant gap.

## 9. Risks (R1 — updated after source read)

| ID | Risk | Likelihood × Impact | Mitigation |
|---|---|---|---|
| R1 | ~~Custom SSM cannot reach parity~~ | ELIMINATED | No SSM exists in ZAYA1. CCA replaces it as the most novel component but is straightforward (no recurrent scan). |
| R1' | CCA depthwise conv with custom groups doesn't match PyTorch numerically | Medium × Medium | Dump CCA submodule outputs at fine granularity (linear_q/k, val_proj1/2, conv_qk[0], conv_qk[1], post-L2-norm Q/K). Test MLX `nn.Conv1d(groups=...)` against reference for a small synthetic input before integration. |
| R2 | Zyphra transformers fork doesn't build on macOS | RESOLVED | Built on first try on M3 Max. |
| R3 | RAM pressure under simultaneous PyTorch + MLX | Medium × Medium | Mitigated by offline-dump workflow |
| R4 | `mlx_lm.convert` mishandles MoE / depthwise conv weights | Medium × Medium | Study `mlx_lm/models/jamba.py` for MoE quantization patterns. Verify quant works correctly on CCA's depthwise convs by spot-checking weights post-quant. Add custom quant-skip predicate for router_mlp's GELU layers if needed. |
| R5 | HF safetensors → MLX weight key mismatches | Low × Medium | `sanitize` logs every unmapped key; tied lm_head weight handled explicitly (don't load it; alias from embed_tokens). |
| R6 | `mlx-community` HF org membership unknown | Low × Low | Check at start of Phase 12; fall back to personal namespace, request access in parallel |
| R7 | 4-bit quant disproportionately damages router / EDA | Low × Medium (unknown) | Phase 11 gate covers; mitigation built into stop conditions |
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

# ZAYA1 Architecture Notes

**Source:** Zyphra/transformers @ zaya1, commit `f0ab5bef`
**Read date:** 2026-05-06
**Files:** `configuration_zaya.py` (126 LOC), `modular_zaya.py` (2,316 LOC, authored), `modeling_zaya.py` (2,069 LOC, auto-generated from modular)

---

## Headline corrections to the project brief

The project brief and design spec described ZAYA1 as a **Mamba+Attention+MoE hybrid** with a "custom SSM" as the highest-risk port. **This is wrong.**

The actual architecture has:

- **No SSM scan.** What looked like a Mamba component is **CCA (Compressed Causal Attention)** — a custom attention variant with two depthwise 1D causal convolutions (kernel size 2) along sequence and a time-shifted value stream. The `mamba_cache_dtype: float32` config flag and the `conv_states` tensor in the cache are conv state, not SSM state.
- **Strict 1:1 alternation** between attention layers (CCA) and MoE layers, not "every layer is hybrid."
- **40 ATT layers + 40 MoE layers** total = 80 layers. ATT on even indices, MoE on odd indices.
- **`ZayaRMSNorm` is identical to standard RMSNorm** (T5/Llama style). Use mlx-lm's `nn.RMSNorm` directly.
- **`ZayaRotaryEmbedding` is identical to `Glm4RotaryEmbedding`** (which inherits from standard RoPE). No custom math.
- **EDA is not exponential averaging.** It's a learnable affine combination of the previous layer's router hidden states with the current layer's, threaded only through MoE layers.

Implication for the spec: **R1 (highest-risk SSM port) is fully eliminated.** Phase 4 of the implementation plan (port the SSM) does not need to exist. CCA replaces it as the most architecturally novel piece, but it's still ordinary linear algebra, not a recurrent scan.

---

## Class catalog (in dependency order)

### 1. `swiglu` / `bias_swiglu` (functions)

Standard SwiGLU activation: `silu(x_1) * x_2` where `x = [x_1, x_2]` chunked along last dim. The fused `BiasSwiGLUFunction` is a perf optimization; the math is the same.

- **MLX porting:** trivial — `mx.sigmoid(x_1) * x_1 * x_2` (mlx has `nn.SiLU`).

### 2. `rotate_half(x)`

Rotates the last-dim halves: returns `cat[-x[..., d/2:], x[..., :d/2]]` along last dim.

- **MLX porting:** trivial.

### 3. `apply_rotary_pos_emb(q, k, cos, sin)` (modular_zaya.py:185-210)

This is the **partial RoPE** implementation:
- `rotary_dim = cos.shape[-1]` — the rotary dimension is inferred from cos/sin tensor shape, NOT taken as an explicit parameter
- Slices q into `q_rot = q[..., :rotary_dim]` and `q_pass = q[..., rotary_dim:]`
- Applies standard RoPE rotation only to `q_rot`: `q_rot * cos + rotate_half(q_rot) * sin`
- Concatenates back: `q_embed = cat[q_rot, q_pass]`

The cos/sin tensors are pre-trimmed by `ZayaRotaryEmbedding` to size `head_dim * partial_rotary_factor`. With `head_dim=128` and `partial_rotary_factor=0.5`, rotary_dim = 64. So 64 dims are rotated, 64 are passed through.

- **MLX porting:** straightforward. mlx-lm's `nn.RoPE(dims=64, traditional=False, base=rope_theta)` plus a slice/concat wrapper.

### 4. `ZayaRotaryEmbedding`

`class ZayaRotaryEmbedding(Glm4RotaryEmbedding): pass` — standard RoPE with `partial_rotary_factor=0.5` baked into the cos/sin computation. The base RoPE math is the same as Llama; the only difference is the rotary_dim is half the head_dim.

- **MLX porting:** use `nn.RoPE(dims=int(head_dim * partial_rotary_factor), base=rope_theta)`. Apply only to the first `rotary_dim` features per head.

### 5. `ZayaRMSNorm` (modeling_zaya.py:166-183, expanded form)

Standard RMSNorm:
```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + eps)
return self.weight * hidden_states.to(input_dtype)
```

- **MLX porting:** `nn.RMSNorm(hidden_size, eps=norm_epsilon)` — drop-in replacement.

### 6. `ZayaDynamicCache` (modular_zaya.py:221-282)

Custom cache class extending `DynamicCache`. Stores per-layer:
- Standard KV cache (inherited)
- `conv_states[layer, batch, in_out_ch, kernel_size=2]` — depthwise conv state for CCA's 1D conv along sequence
- `prev_hs[layer, batch, hidden_size]` — last hidden state from layer's input, for CCA's V₂ (time-shifted) stream during generation

Methods: `update_conv_state` (rolls + replaces last slot), `reset` (zero everything).

- **MLX porting:** subclass of mlx-lm's cache base. The conv_states and prev_hs need explicit allocation upfront because the `num_layers` is fixed and per-layer state shape is uniform.

### 7. `CCA` (modular_zaya.py:285-521) — **the most novel attention component**

CCA = Compressed Causal Attention. Replaces standard QKV projection with a learned mechanism:

**Heads:** `cca_num_q_heads=8`, `cca_num_kv_heads=2`, `head_dim=128`. Q is 8×128=1024, K and V are each 2×128=256.

**Init parameters:**
- `linear_q: Linear(2048, 1024, bias=False)` — Q projection
- `linear_k: Linear(2048, 256, bias=False)` — K projection
- `val_proj1: Linear(2048, 128, bias=False)` — first V stream (uses current hidden state)
- `val_proj2: Linear(2048, 128, bias=False)` — second V stream (uses time-shifted hidden state)
- `conv_qk: Sequential(Conv1d, Conv1d)` — two-stage 1D causal conv on concatenated [Q, K]:
  - First conv: `kernel=2`, `groups=in_out_ch=1280` (depthwise, fully separable)
  - Second conv: `kernel=2`, `groups=num_kv_heads + num_q_heads = 10` (grouped per-head)
- `temp: Parameter(torch.zeros(num_kv_heads=2))` — per-KV-head learnable temperature

**Forward** (modular_zaya.py:370-521):

Layout note: forward switches between HF `[B, S, H]` and `[S, B, H]` layouts mid-function. MLX port should pick one consistent layout.

Computation steps:
1. `q = linear_q(hs)`, `k = linear_k(hs)` — both from current hidden state
2. `qk_packed0 = cat([q, k], dim=-1)` — shape `[S, B, 1280]`
3. **Pre-conv mean residual**:
   - `query_pre` reshape q to `[S, B, 8, 128]`
   - `key_pre` reshape k to `[S, B, 2, 128]`, then repeat 4× along head axis → `[S, B, 8, 128]`
   - `qk_mean_q = (query_pre + key_pre) / 2`
   - `qk_mean_k = qk_mean_q.view(S, B, 2, 4, 128).mean(dim=-2)` → `[S, B, 2, 128]`
4. **Two-stage causal conv** on `qk_packed0` along sequence:
   - Pad with `total_padding = (cca_time0 - 1) + (cca_time1 - 1) = 2` zeros at the front
   - Apply `conv_qk` → `qk_packed3` shape `[S, B, 1280]`
5. `query = qk_packed3[..., :1024].view(S, B, 8, 128) + qk_mean_q`
6. `key = qk_packed3[..., 1024:].view(S, B, 2, 128) + qk_mean_k`
7. **V from two streams**:
   - `v1 = val_proj1(hs)` (current step)
   - `v2 = val_proj2(hs_d)` where `hs_d = pad(hs[:-1], front by 1)` (one step earlier; in generation, pulled from `prev_hs[layer]`)
   - `value = cat([v1, v2], dim=-1).view(S, B, 2, 128)`
8. **L2 normalize Q and K, apply per-head temperature**:
   - `query = query * (sqrt(head_dim) / ||query||₂_perhead)`
   - `key = key * (sqrt(head_dim) / ||key||₂_perhead) * temp[None, None, :, None]`
9. Reshape and transpose back to HF layout `[B, S, num_*heads * head_dim]`

Returns `(query, key, value)` ready for standard attention computation.

**MLX porting notes:**
- The two depthwise convs are the trickiest piece; mlx has `nn.Conv1d` but groups+depthwise behavior needs verification on MLX.
- During prefill: pad once at front, conv along sequence, get full output.
- During generation (single token): cache the last `kernel_size` Q+K values, concatenate with new step, conv, advance cache.
- The per-head temperature initialized to zeros means at init, K is multiplied by 0 — this is unusual and may indicate Zyphra wants the model to learn the temperature from scratch starting at zero (similar to how some QK-norm impls start with very small init).

### 8. `ZayaAttention` (modular_zaya.py:524-656)

Wraps CCA + standard scaled dot product attention:

- `self.qkv = CCA(...)` — produces Q, K, V
- `self.o_proj = Linear(num_heads // 2 * head_dim = 1024, hidden_size = 2048)` — output projection from compressed (8-head) attention back to hidden_size
- Forward: `query_states, key_states, value_states = self.qkv(...)`, reshape to multi-head, transpose to `[B, H, S, D]`, apply RoPE via `apply_rotary_pos_emb` (with cos/sin from `ZayaRotaryEmbedding`), update KV cache, `repeat_kv(k_or_v, num_key_value_groups // 2 = 4)` for GQA, standard `Q @ K.T / sqrt(d)`, softmax in fp32, `attn @ V`, transpose, view as `(B, S, hidden_size // 2 = 1024)`, then `o_proj` back to 2048.

**Notable:** the eager forward upcasts softmax to fp32 then back to query dtype. The flash and SDPA variants exist but config defaults to `eager`.

- **MLX porting:** standard attention pattern. Use mlx-lm's `nn.scaled_dot_product_attention` or manual implementation. Match the eager path including fp32 softmax.

### 9. `ZayaDecoderATTLayer` (modular_zaya.py:909-1000) — **even-indexed decoder layers**

Composes:
- `self_attn: ZayaAttention(config, layer_n)`
- `input_norm: ZayaRMSNorm(hidden_size, eps=norm_epsilon)`
- `res_scale: ResidualScaling(config, layer_n)` (optional, present when `scale_residual_merge=True`)

Forward signature: `(hidden_states, residual, attention_mask, cca_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, prev_router_hidden_states, **kwargs)` → returns `(outputs, residual, prev_router_hidden_states)`.

Computation:
1. `if scale_residual_merge: residual, hidden_states = res_scale(residual, hidden_states)` — affine on both streams
2. If `residual is None`: `residual = hidden_states (in fp32 if residual_in_fp32)` — first layer initializes residual from hidden_states
3. Else: `residual = hidden_states + residual` — accumulate
4. `hidden_states = input_norm(residual)` — pre-norm based on accumulated residual
5. `hidden_states = self_attn(hidden_states, ...)`
6. Returns `(hidden_states,) + (attn_weights,)?, residual, prev_router_hidden_states` — passes `prev_router_hidden_states` through unchanged (this layer is not a router)

Key insight: **the residual stream is separate from hidden_states.** Each block writes to hidden_states; residual is updated *before* the next block (by the next layer's residual+hidden_states merge). This is non-standard pre-norm — most pre-norm transformers do `hs = hs + block(norm(hs))` per block, but ZAYA1 does `residual += hs; hs = block(norm(residual))`.

### 10. `ResidualScaling` (modular_zaya.py:1003-1033)

Per-feature affine on the residual streams:
- `hidden_states_scale: Parameter(ones(hidden_size))`, `hidden_states_bias: Parameter(zeros(hidden_size))`
- For non-first-layer also: `residual_scale: Parameter(ones(hidden_size))`, `residual_bias: Parameter(zeros(hidden_size))`

Forward:
```python
hidden_states = (hidden_states + hidden_states_bias) * hidden_states_scale
if not_first_layer:
    residual = (residual + residual_bias) * residual_scale
return residual, hidden_states
```

- **MLX porting:** trivial. Two `nn.Linear` substitutes (with diagonal weights) or just direct elementwise ops.

### 11. `ZayaRouter` (modular_zaya.py:1036-1187) — the MoE router with EDA

**Init parameters:**
- `down_proj: Linear(hidden_size, mlp_expansion=256, bias=True)`
- `rmsnorm_eda: ZayaRMSNorm(mlp_expansion=256, eps=norm_epsilon)`
- `router_states_scale: Parameter(ones(mlp_expansion))` — only when `use_eda=True` (i.e. layer_number != 1)
- `router_mlp: Sequential(Linear(D, D), GELU, Linear(D, D), GELU, Linear(D, E, bias=False))` — three-layer MLP
- `balancing_biases: Buffer(zeros(num_experts))` — for load balancing; if `use_mod`: `balancing_biases[-1] = -1.0` initially
- `use_mod=True`: `num_experts = num_moe_experts + 1 = 17` (16 real experts + 1 skip expert)
- `topk=1`

**EDA gate:** `use_eda = use_eda_cfg AND (zaya_first_layer is not None) AND (layer_number != zaya_first_layer)` where `zaya_first_layer = 1` (hardcoded in __init__).

So EDA is enabled in the router for **all MoE layers except the very first MoE layer (layer 1 in the global index, which is the first MoE layer)**. This means:
- Layer 1 (first MoE): no EDA
- Layer 3, 5, 7, ..., 79 (subsequent MoE layers): EDA active

**Forward** (line 1125-1187):

Input: `hidden_states (B, S, H), router_states (B, S, D) — previous MoE layer's pre-norm router hidden states`.

Steps:
1. `hs = down_proj(hidden_states)` → `(B, S, 256)`
2. If `use_eda` and `router_states is not None`: `hs = hs + router_states * router_states_scale` — EDA
3. `router_hidden_states_next = hs[:, -S:].clone()` — stash pre-norm post-EDA hs for the *next* MoE layer
4. `hs_norm = rmsnorm_eda(hs)`
5. `logits = router_mlp(hs_norm)` → `(B, S, num_experts=17)`
6. `expert_prob = softmax(logits, dim=-1)`
7. `biased = expert_prob.detach().to(fp32) + balancing_biases` — balancing biases affect selection but not the gradient
8. `_, expert_choice_t = topk(biased, topk=1, dim=-1)` → `(B, S, 1)`
9. (If topk>1 and use_mod: cumulative skip-expert mask propagation. With topk=1 this branch is dead.)
10. `route_prob = gather(expert_prob, dim=2, index=expert_choice_t)` → `(B, S, 1)`
11. Returns `(route_prob_flat (B*S, 1), expert_choice_flat (B*S, 1), router_hidden_states_next (B, S, 256))`

**Note:** balancing_biases starts at zeros for all 16 real experts and -1.0 for the skip expert. Trained values would have evolved away from these defaults.

### 12. `MLP` (modular_zaya.py:1190-1275)

A single SwiGLU MLP (one expert). Standard:
- `linear_fc1: Linear(hidden_size=2048, ffn_hidden_size=4096, bias=False)` — note ffn_hidden_size=4096, with gated linear unit the *output* width is halved to 2048 effective
- `linear_fc2: Linear(ffn_hidden_size_out=2048, hidden_size=2048, bias=False)`
- Forward: `linear_fc1` → split in half → `silu(half_a) * half_b` → `linear_fc2`

`add_bias_linear=False` per config, so no bias terms.

### 13. `SequentialMLP` (modular_zaya.py:1278-1326)

Container for MoE experts. Holds `local_experts: ModuleList[MLP]` of length `num_moe_experts`. Forward gets `(permuted_local_hidden_states, tokens_per_expert)` — tokens already routed to their experts and concatenated, with a count of how many tokens go to each expert.

Splits the input by `tokens_per_expert.tolist()`, passes each chunk through its corresponding expert, concatenates outputs.

If `num_local_experts == 1`: shortcut to single expert.

### 14. `ZayaBlock` (modular_zaya.py:1329-1422) — the MoE forward

Combines router + experts + MoD logic.

**Forward:**
1. `route_prob, expert_choice, prev_router_hidden_states = self.router(hidden_states, router_states=prev_router_hidden_states)`
2. Flatten hs to `(B*S, H)`, flatten indices to `(B*S,)`
3. Sort by expert index: `sorted_indices, sort_order = torch.sort(indices_flat)`
4. `tokens_per_expert = bincount(sorted_indices, minlength=num_experts=17)`
5. `sorted_hidden_states = hidden_states_flat[sort_order]` (gather)
6. `original_order = argsort(sort_order)` (for un-permuting later)
7. **MoD branch (`zaya_use_mod=True`):**
   - The skip expert is the LAST one (index 16). Tokens routed to it should bypass all MLPs.
   - Run `experts(sorted_hidden_states[:sum(tokens_per_expert[:-1])], tokens_per_expert[:-1])` — only run the first 16 (real) experts
   - For the tokens routed to skip expert: `sorted_hidden_states[sum(tokens_per_expert[:-1]):]` — these are passed through unchanged
   - Concatenate back: `expert_output = cat([real_outputs, skip_passthroughs])`
8. Else (no MoD): just run all experts.
9. `expert_output = expert_output[original_order]` — un-permute back to original sequence order
10. `expert_output = expert_output.view(B, S, H)`
11. `expert_output = expert_output * route_prob.unsqueeze(-1)` — scale by gating probability
12. Returns `(expert_output, mlp_bias_or_None, prev_router_hidden_states)`

**Insight about MoD:** with top-1 routing, exactly one expert is chosen per token, and that one expert can be the skip expert (index 16). When chosen, the token goes through unchanged but with its gating probability applied (multiplicatively). Since the gating prob for skip is the softmax output, it is a small positive number, which reduces the magnitude of the output for skipped tokens. This is functionally similar to a learnable gate that decides "should I skip this token's MLP."

### 15. `ZayaDecoderMLPLayer` (modular_zaya.py:1425-1533) — **odd-indexed decoder layers**

Symmetric to `ZayaDecoderATTLayer`:
- `zaya_block: ZayaBlock(...)` — the MoE
- `input_norm: ZayaRMSNorm(...)`
- `res_scale: ResidualScaling(...)` (optional)

Forward: same residual+norm pattern as ATT layer, then `zaya_block` instead of `self_attn`. Returns `(hidden_states,), residual, prev_router_hidden_states` — passes residual through, **updates** prev_router_hidden_states from the router.

### 16. `ZayaModel` (modular_zaya.py:1642-1956)

The full model:
- `embed_tokens: Embedding(vocab_size=262272, hidden_size=2048)`
- `layers: ModuleList[ZayaDecoderATTLayer or ZayaDecoderMLPLayer]` of length 80, alternating ATT/MoE per `layer_n % 2`
- `res_scale: ResidualScaling(config, num_hidden_layers=80)` — final residual scaling (used at exit, after the last layer)
- `final_norm: ZayaRMSNorm(...)`
- `rotary_emb: ZayaRotaryEmbedding(...)`

Forward (line 1700+):
1. Compute `cca_mask = attention_mask.clone()` if attention mask given
2. Embed: `hidden_states = embed_tokens(input_ids)`
3. `residual = None`, `prev_router_hidden_states = None`
4. Compute `position_embeddings = rotary_emb(hidden_states, position_ids)` — cos/sin
5. Compute causal mask via `_update_causal_mask`
6. For each layer: `(layer_outputs, residual, prev_router_hidden_states) = layer(hidden_states, residual, ..., prev_router_hidden_states)`; `hidden_states = layer_outputs[0]`
7. Final residual merge: `if scale_residual_merge: residual, hidden_states = res_scale(residual, hidden_states); residual = hidden_states + residual; hidden_states = final_norm(residual)`
8. Returns `MoeModelOutputWithPast(last_hidden_state=hidden_states, ...)`

### 17. `ZayaForCausalLM` (modular_zaya.py:2048-2302)

Wraps `ZayaModel` + `lm_head`:
- `model: ZayaModel(config)`
- `lm_head: Linear(hidden_size, vocab_size, bias=lm_head_bias=False)`
- **Tied weights:** `if tie_word_embeddings: self.lm_head.weight = self.model.embed_tokens.weight` (line 2059-2060). For our `sanitize(weights)` in mlx-lm, we don't load `lm_head.weight` separately — it's the same tensor as `embed_tokens.weight`.

Forward: just `model(...)` then `lm_head(hidden_states[..., slice_indices])`.

---

## Open question answers (from spec §5)

### Q1: Layer schedule — interleaved or hybrid?

**INTERLEAVED.** modular_zaya.py:1663-1681:
```python
for layer_n in range(config.num_hidden_layers):
    if layer_n % 2 == 1:
        self.layers.append(ZayaDecoderMLPLayer(...))
    else:
        self.layers.append(ZayaDecoderATTLayer(...))
```

Even layers (0, 2, 4, ..., 78) = ATT (CCA attention). Odd layers (1, 3, 5, ..., 79) = MoE.

### Q2: MoE coverage — every layer or subset?

**40 of 80 layers (every odd layer).** Same code as Q1.

### Q3: ZayaRMSNorm semantics — different from stock?

**Identical to standard RMSNorm.** modular_zaya.py:217-218: `class ZayaRMSNorm(LlamaRMSNorm): pass`. modeling_zaya.py:166-183 expanded form is bog-standard RMSNorm: variance in fp32, `x * rsqrt(var + eps) * weight`, no bias.

### Q4: scale_residual_merge formula

**Per-feature affine on both residual streams before merging.** modular_zaya.py:1024-1033:
```python
hidden_states = (hidden_states + hs_bias) * hs_scale
if not_first_layer:
    residual = (residual + res_bias) * res_scale
return residual, hidden_states
```
The first layer skips the residual transform (no residual yet). All four parameters (hs_scale, hs_bias, res_scale, res_bias) are per-feature `Parameter(shape=(hidden_size,))`. Init: scales to ones, biases to zeros.

### Q5: EDA exact form

**Learnable per-feature affine combination of consecutive MoE layers' router hidden states, applied INSIDE the router, NOT exponential averaging.**

modular_zaya.py:1153-1154:
```python
if self.use_eda and (router_states is not None):
    hs = hs + router_states * self.router_states_scale
```

`router_states` is the previous MoE layer's `router_hidden_states_next` — defined as the post-down-projection, post-EDA, pre-RMSNorm hs (line 1157: `router_hidden_states_next = hs[:, -S:].clone()`). `router_states_scale` is a `Parameter(ones(mlp_expansion=256))`.

So EDA is recursively cumulative: each MoE layer's pre-norm router hs encodes information from all prior MoE layers' router hs, weighted by the per-layer learnable scale. Definitely not exponential — there's no decay factor and no fixed weighting.

EDA is gated off for layer 1 (the first MoE layer) by hardcoded `zaya_first_layer = 1`.

---

## Forward-pass call graph (decoder layer level)

```
ZayaForCausalLM.forward
└── ZayaModel.forward
    ├── embed_tokens(input_ids) → hidden_states
    ├── rotary_emb(hidden_states, position_ids) → (cos, sin)
    └── for layer_n in 0..79:
        ├── if layer_n even: ZayaDecoderATTLayer.forward
        │   ├── res_scale(residual, hidden_states)
        │   ├── residual = hidden_states + residual
        │   ├── hidden_states = input_norm(residual)
        │   ├── ZayaAttention.forward
        │   │   ├── CCA.forward
        │   │   │   ├── linear_q, linear_k, val_proj1, val_proj2
        │   │   │   ├── conv_qk[Conv1d, Conv1d] (depthwise causal conv)
        │   │   │   ├── L2-normalize Q, K + apply temp
        │   │   │   └── return Q, K, V
        │   │   ├── apply_rotary_pos_emb(q, k, cos, sin)  ← partial RoPE here
        │   │   ├── past_key_values.update(k, v)          ← KV cache
        │   │   ├── repeat_kv(k, v, group_size=4)         ← GQA
        │   │   ├── softmax(QK.T / sqrt(d)) in fp32
        │   │   └── o_proj(attn @ V)
        │   └── return (hidden_states,), residual, prev_router_hidden_states
        └── if layer_n odd: ZayaDecoderMLPLayer.forward
            ├── res_scale(residual, hidden_states)
            ├── residual = hidden_states + residual
            ├── hidden_states = input_norm(residual)
            ├── ZayaBlock.forward
            │   ├── ZayaRouter.forward
            │   │   ├── down_proj(hs) → (B, S, 256)
            │   │   ├── if use_eda: hs += prev_router_states * router_states_scale
            │   │   ├── stash router_hidden_states_next
            │   │   ├── rmsnorm_eda(hs)
            │   │   ├── router_mlp(hs_norm) → (B, S, 17)
            │   │   ├── softmax + balancing_biases + topk(1)
            │   │   └── return (route_prob, expert_choice, router_hidden_states_next)
            │   ├── permute by expert_choice, bincount tokens_per_expert
            │   ├── SequentialMLP.forward (16 experts; skip expert=17 bypassed)
            │   ├── un-permute, gate by route_prob
            │   └── return expert_output, mlp_bias, prev_router_hidden_states
            └── return (hidden_states,), residual, prev_router_hidden_states
    └── final residual merge + final_norm
```

---

## Submodules to hook in dump_activations.py

For each layer, hook every submodule that has weights or produces a meaningful intermediate. Naming convention: `L{layer_idx}_{leaf_name}_out` for forward-hook outputs.

Below is the canonical hook list. **Even-indexed layers** are ATT (use the `att_*` keys). **Odd-indexed layers** are MoE (use the `moe_*` keys).

### Common to all layers

| key | Description |
|---|---|
| `input_norm_out` | Output of pre-block RMSNorm |
| `res_scale_residual_out` | Residual stream after ResidualScaling (if scale_residual_merge=True) |
| `res_scale_hs_out` | Hidden states after ResidualScaling |
| `layer_out` | Layer's output hidden_states (after the block) |

### Even layers (ATT)

| key | Description |
|---|---|
| `self_attn_qkv_linear_q_out` | CCA Q projection |
| `self_attn_qkv_linear_k_out` | CCA K projection |
| `self_attn_qkv_val_proj1_out` | CCA V₁ stream |
| `self_attn_qkv_val_proj2_out` | CCA V₂ stream (time-shifted) |
| `self_attn_qkv_conv_qk_0_out` | First depthwise conv output |
| `self_attn_qkv_conv_qk_1_out` | Second depthwise conv output (final qk_packed3) |
| `self_attn_qkv_out` | Tuple (Q, K, V) — capture as a triplet, save Q only as `qkv_q`, K as `qkv_k`, V as `qkv_v` |
| `self_attn_o_proj_out` | Output projection of attention |
| `self_attn_out` | Final attention output (= o_proj output here) |

### Odd layers (MoE)

| key | Description |
|---|---|
| `zaya_block_router_down_proj_out` | Router down projection |
| `zaya_block_router_rmsnorm_eda_out` | Pre-MLP norm |
| `zaya_block_router_router_mlp_0_out` | First router MLP linear |
| `zaya_block_router_router_mlp_2_out` | Second router MLP linear |
| `zaya_block_router_router_mlp_4_out` | Third router MLP linear (logits) |
| `zaya_block_router_out` | Tuple (route_prob, expert_choice, router_hidden_states_next) — capture each |
| `zaya_block_experts_local_experts_{i}_linear_fc1_out` | Per-expert FC1 (typically only one is fired per token) |
| `zaya_block_experts_local_experts_{i}_linear_fc2_out` | Per-expert FC2 |
| `zaya_block_out` | MoE block output (post gating) |

### Model level

| key | Description |
|---|---|
| `embed_tokens_out` | Token embedding output |
| `rotary_emb_out` | (cos, sin) tuple — capture the two tensors as `cos`/`sin` |
| `final_residual_out` | Residual after final ResidualScaling+merge |
| `final_norm_out` | Output of final_norm (pre-lm_head input) |
| `lm_head_out` | Logits |

---

## Captured shape inventory

To be filled in by Task 8 once the dump runs. Expected key counts:
- Common per layer × 80 layers = 320 entries
- Even-layer ATT: 9 keys × 40 layers = 360 entries
- Odd-layer MoE: ~4 + 16×2 = 36 keys per layer × 40 layers = 1,440 entries (most experts will be empty for any given prompt — only chosen experts produce output)
- Model level: ~5 entries

Total: ~2,000 .npy files per prompt. A 32-token forward pass on smoke prompt with 80 layers will produce roughly this many tensor outputs.

---

## Implications for spec §5 facts

The original spec §5 listed these architectural components:

| Spec claim | Reality | Action |
|---|---|---|
| Custom SSM / Mamba-style layers | **No SSM. CCA (depthwise conv + time-shifted V) on even layers.** | Update spec, drop "Phase 4 SSM port" |
| MoE with top-1 routing on every layer | **Only on odd layers (40 of 80).** | Update spec |
| MoD via skip expert | **Confirmed.** Skip expert is index 16 of 17 (16 real + 1 skip). Tokens routed to skip pass through unchanged but multiplied by gating prob. | Keep |
| EDA on hidden states | **Actually on router hidden states (per-feature 256-dim tensor), not main 2048-dim hidden states. Affine combination, not exponential.** | Update spec |
| Partial RoPE | **Confirmed.** rotary_dim = head_dim × 0.5 = 64. | Keep |
| ZayaRMSNorm custom | **Standard.** Use mlx-lm `nn.RMSNorm`. | Simplify |
| `scale_residual_merge` formula | **Per-feature affine on both streams: `(stream + bias) * scale`.** | Update spec |
| `mamba_cache_dtype: float32` | **Misnamed — applies to CCA conv state, not an SSM state.** | Update spec |
| `residual_in_fp32: True` | **Confirmed.** First-layer residual init uses fp32. | Keep |

---

## Updated risk register

| Original risk | Status |
|---|---|
| R1 (custom SSM parity unreachable) | **Eliminated.** No SSM exists. CCA replaces it but is straightforward depthwise conv + time-shift, not a recurrent scan. |
| R2 (Zyphra fork doesn't build on Mac) | **Resolved.** Builds first try. |
| R4 (`mlx_lm.convert` mishandles MoE/SSM weights) | **Reduced.** No SSM. Just MoE expert MLPs (standard linears) + the depthwise convs in CCA — quantization should handle these naturally. |
| New: CCA depthwise conv groups | Small risk: MLX's `nn.Conv1d` group support needs verification. Mitigation: a unit test against the saved `conv_qk_*_out` activations. |
| New: per-head L2 normalization with `temp[None, None, :, None]` | Small: confirm broadcasting on 4D query tensor. |

---

## Implications for the Phase 0 hook taxonomy

The hook taxonomy in `2026-05-06-phase0-reference-scaffolding.md` (Task 6) was speculative and based on the wrong architecture model. **Task 6 should be regenerated using this section's hook list before Task 7 starts.**

## Implications for the implementation plan structure

The original 9-phase plan had:
- Phase 2: ZayaRMSNorm + partial RoPE
- Phase 3: Attention
- Phase 4: SSM ← **delete this phase**
- Phase 5: MoE + MoD + EDA

Updated phase plan:
- Phase 2: partial RoPE + (no work for ZayaRMSNorm; use stock)
- Phase 3: CCA (the longest of the new phases — depthwise conv + time-shift + L2 norm)
- Phase 4: ZayaAttention wrapping CCA + standard scaled dot product
- Phase 5: ResidualScaling + ZayaDecoderATTLayer
- Phase 6: ZayaRouter (with EDA) + SequentialMLP + ZayaBlock + MoD logic
- Phase 7: ZayaDecoderMLPLayer
- Phase 8: ZayaModel forward (residual threading + alternation)
- Phase 9: ZayaForCausalLM + tied embeddings + sanitize
- Phase 10: end-to-end forward parity + greedy decode parity
- Phase 11: 4-bit quantization
- Phase 12: HF upload + Zyphra issue

This needs to be reflected in a revised spec.

---

## What I did NOT have to read carefully

- The HF causal mask preparation (`_update_causal_mask`, `_prepare_4d_causal_attention_mask_with_cache_position`) — these are vanilla copies from Phi3/Mistral. mlx-lm's standard mask creation suffices.
- The attention impl variants `ZayaSdpaAttention` and `ZayaFlashAttention2` — config defaults to `eager`. We port only the eager path.
- The `ZayaForSequenceClassification` — out of scope for our causal-LM port.
- `prepare_inputs_for_generation` — out of scope; mlx-lm's generation utilities handle this.

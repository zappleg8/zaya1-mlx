# HF safetensors weight key inventory: Zyphra/ZAYA1-8B

**Source:** `model.safetensors.index.json` from the HF snapshot.
**Total tensors:** 2,483
**Total bf16 bytes:** 17,680,978,928 (~16.5 GiB)
**Approx params (bf16):** 8,840,489,464 (~8.84 B)

## Top-level keys (6 total)

- `model.embed_tokens.weight` — embedding table, shape (vocab_size=262272, hidden_size=2048)
- `model.final_norm.weight` — final RMSNorm gain, shape (hidden_size=2048,)
- `model.res_scale.hidden_states_bias` — final ResidualScaling, shape (hidden_size,)
- `model.res_scale.hidden_states_scale` — final ResidualScaling, shape (hidden_size,)
- `model.res_scale.residual_bias` — final ResidualScaling, shape (hidden_size,)
- `model.res_scale.residual_scale` — final ResidualScaling, shape (hidden_size,)

**`lm_head.weight` is NOT in the safetensors** — the model uses `tie_word_embeddings=True`, so lm_head is aliased to embed_tokens at load time.

## Per-layer keys

### Even layers (0, 2, 4, …, 78) — ATT layer

Layer 0 has 13 keys (no `res_scale.residual_*` because it's the first layer).
Other even layers have 15 keys.

```
model.layers.{i}.input_norm.weight
model.layers.{i}.res_scale.hidden_states_bias
model.layers.{i}.res_scale.hidden_states_scale
model.layers.{i}.res_scale.residual_bias        # only if i != 0
model.layers.{i}.res_scale.residual_scale       # only if i != 0
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.self_attn.qkv.linear_q.weight
model.layers.{i}.self_attn.qkv.linear_k.weight
model.layers.{i}.self_attn.qkv.val_proj1.weight
model.layers.{i}.self_attn.qkv.val_proj2.weight
model.layers.{i}.self_attn.qkv.conv_qk.0.weight
model.layers.{i}.self_attn.qkv.conv_qk.0.bias
model.layers.{i}.self_attn.qkv.conv_qk.1.weight
model.layers.{i}.self_attn.qkv.conv_qk.1.bias
model.layers.{i}.self_attn.qkv.temp
```

### Odd layers (1, 3, 5, …, 79) — MoE layer

Layer 1 has 46 keys (no `router_states_scale` because EDA is gated off for the first MoE layer).
Other odd layers have 47 keys.

```
model.layers.{i}.input_norm.weight
model.layers.{i}.res_scale.hidden_states_bias
model.layers.{i}.res_scale.hidden_states_scale
model.layers.{i}.res_scale.residual_bias        # always present for odd layers
model.layers.{i}.res_scale.residual_scale
model.layers.{i}.zaya_block.router.balancing_biases     # buffer in PyTorch; loaded as mx.array in MLX
model.layers.{i}.zaya_block.router.down_proj.weight
model.layers.{i}.zaya_block.router.down_proj.bias
model.layers.{i}.zaya_block.router.rmsnorm_eda.weight
model.layers.{i}.zaya_block.router.router_states_scale  # only if i != 1 (EDA)
model.layers.{i}.zaya_block.router.router_mlp.0.weight
model.layers.{i}.zaya_block.router.router_mlp.0.bias
model.layers.{i}.zaya_block.router.router_mlp.2.weight
model.layers.{i}.zaya_block.router.router_mlp.2.bias
model.layers.{i}.zaya_block.router.router_mlp.4.weight  # NB: no bias on the output linear
model.layers.{i}.zaya_block.experts.local_experts.{e}.linear_fc1.weight  # for e in 0..15
model.layers.{i}.zaya_block.experts.local_experts.{e}.linear_fc2.weight  # for e in 0..15
```

## Shape reference

| key | shape | notes |
|---|---|---|
| embed_tokens.weight | (262272, 2048) | tied to lm_head; only one copy on disk |
| final_norm.weight | (2048,) | RMSNorm gain |
| input_norm.weight | (2048,) | per-layer pre-block norm |
| res_scale.hidden_states_scale | (2048,) | diagonal affine, per-feature |
| res_scale.hidden_states_bias | (2048,) | per-feature bias |
| res_scale.residual_scale | (2048,) | only for non-first layer |
| res_scale.residual_bias | (2048,) | only for non-first layer |
| self_attn.o_proj.weight | (2048, 1024) | input dim is hidden_size//2 (CCA query compression: 8 q heads × 128) |
| self_attn.qkv.linear_q.weight | (1024, 2048) | 8 q heads × 128 dim |
| self_attn.qkv.linear_k.weight | (256, 2048) | 2 kv heads × 128 dim |
| self_attn.qkv.val_proj1.weight | (128, 2048) | latent_k_dim/2 = 256/2 |
| self_attn.qkv.val_proj2.weight | (128, 2048) | latent_k_dim/2 |
| self_attn.qkv.conv_qk.0.weight | (1280, 1, 2) | depthwise (groups=1280), kernel=2; PyTorch shape = (out, in/g, k) |
| self_attn.qkv.conv_qk.0.bias | (1280,) | |
| self_attn.qkv.conv_qk.1.weight | (1280, 128, 2) | grouped (groups=10), kernel=2; in/groups = 1280/10 = 128 |
| self_attn.qkv.conv_qk.1.bias | (1280,) | |
| self_attn.qkv.temp | (2,) | per-KV-head learnable temperature |
| zaya_block.router.balancing_biases | (17,) | num_experts+1 (16 real + 1 skip) |
| zaya_block.router.down_proj.weight | (256, 2048) | hidden_size → mlp_expansion |
| zaya_block.router.down_proj.bias | (256,) | |
| zaya_block.router.rmsnorm_eda.weight | (256,) | RMSNorm gain on mlp_expansion dim |
| zaya_block.router.router_states_scale | (256,) | EDA per-feature gain (39 of 40 MoE layers) |
| zaya_block.router.router_mlp.0.weight | (256, 256) | first router MLP linear |
| zaya_block.router.router_mlp.0.bias | (256,) | |
| zaya_block.router.router_mlp.2.weight | (256, 256) | second router MLP linear |
| zaya_block.router.router_mlp.2.bias | (256,) | |
| zaya_block.router.router_mlp.4.weight | (17, 256) | output is num_experts+1 (16 + skip); no bias |
| zaya_block.experts.local_experts.{e}.linear_fc1.weight | (4096, 2048) | ffn_hidden_size; SwiGLU input is full ffn |
| zaya_block.experts.local_experts.{e}.linear_fc2.weight | (2048, 2048) | ffn_hidden_size_out = ffn/2 due to gated linear unit |

**Conv1d weight conventions:** PyTorch's `Conv1d(in_ch, out_ch, kernel, groups=g)` weight shape is `(out_ch, in_ch/groups, kernel_size)`. For `groups=in_ch=out_ch=1280` (depthwise), `in_ch/groups = 1`. For `groups=10`, `in_ch/groups = 1280/10 = 128`.

**MLX Conv1d layout differs:** MLX `nn.Conv1d` weight shape is `(out_channels, kernel_size, in_channels/groups)` — kernel and `in/groups` are swapped. The `sanitize` method in zaya.py must transpose conv_qk weights via `v.transpose(0, 2, 1)`.

## Counts summary

- 40 ATT layers: 1 × 13 keys (layer 0) + 39 × 15 keys = 598 keys
- 40 MoE layers: 1 × 46 keys (layer 1) + 39 × 47 keys = 1,879 keys
- Top-level: 6 keys
- **Total: 598 + 1,879 + 6 = 2,483 ✓ (exact match with safetensors index)**

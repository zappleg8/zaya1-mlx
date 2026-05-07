# mlx-lm conventions for new model ports

Based on reading: `mlx_lm/models/{base.py, llama.py, jamba.py}` in the fork at `~/code/personal/mlx-lm`.

## File structure

A model lives in a single file `mlx_lm/models/<name>.py` with these top-level definitions:

- `@dataclass class ModelArgs(BaseModelArgs):` — config from `config.json`. Field names match the JSON keys exactly. `BaseModelArgs` from `mlx_lm.models.base` provides a `from_dict(cls, params)` classmethod that filters out unknown keys via `inspect.signature(cls).parameters`.
- N module classes (Attention, MLP, etc.) — each `class X(nn.Module)` with `__init__(self, args: ModelArgs)` and `__call__(...)`.
- `class <ModelName>(nn.Module):` — the embedding + layers + final norm wrapper (e.g. `LlamaModel`). Conventionally named `<Name>Model`.
- `class Model(nn.Module):` — the canonical mlx-lm name; wraps the inner model + lm_head; this is what `mlx_lm.load` instantiates. **Always named `Model`**, not `ZayaForCausalLM` or similar.

## Required `Model` class members

From `llama.py`:

- `self.args` (the `ModelArgs` instance)
- `self.model_type` (string, must match the config.json `model_type` field)
- `self.model` (the inner `<Name>Model` with embedding + layers + final_norm)
- `self.lm_head` — only defined when `not args.tie_word_embeddings`
- `__call__(self, inputs: mx.array, cache=None, input_embeddings=None) -> mx.array` returning logits
- `sanitize(self, weights: dict) -> dict` — remap HF weight keys; called by `mlx_lm.load`
- `@property layers` — returns the decoder layer list (`return self.model.layers`)
- optional `make_cache(self)` — returns the per-layer KV cache list

## Tied-word-embeddings: Llama style

From `llama.py:200-220`:

```python
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights
```

**Use this pattern for ZAYA1.** No `lm_head` attribute when tied; use `embed_tokens.as_linear(...)` to project to vocab.

## `nn.Embedding.as_linear`

Documented behavior: applies the embedding's weight as a linear projection. Equivalent to `out @ embed_tokens.weight.T` but implementations may use a more efficient kernel.

## Layer lists

`self.layers = [TransformerBlock(...) for ... in ...]` works as a Python list. mlx-lm walks the tree via mlx's tree utilities, and key paths use list-index notation (`model.layers.0.self_attn.q_proj.weight`). HF naming matches this directly.

## MoE: SwitchGLU vs per-expert lists

mlx-lm has `SwitchGLU` (used in `jamba.py:235-245`), which stacks per-expert weights along a leading dim and dispatches via gather. For ZAYA1 Phase 1 we use a **per-expert list of `MLP` instances**, matching HF naming exactly. This is less efficient at runtime but simpler for skeleton + weight loading. Phase 6+ may convert to `SwitchGLU` for performance after parity is established.

## sanitize patterns

`sanitize` runs before `model.load_weights`. Common transforms:

1. Drop unused HF keys (e.g., `lm_head.weight` if tied + we use `as_linear`).
2. Rename HF → MLX naming if any divergence (often none for new ports — match HF naming).
3. Reshape weights when MLX expects a different layout than PyTorch.

For Zaya: we choose MLX names that mirror HF exactly. Sanitize:
- Pops `lm_head.weight` (a no-op since HF doesn't have it for ZAYA1, but defensive).
- Transposes Conv1d weights from PyTorch `(out, in/g, k)` to MLX `(out, k, in/g)`.

## Conv1d layout difference

PyTorch `nn.Conv1d`: weight shape `(out_channels, in_channels // groups, kernel_size)`.

MLX `nn.Conv1d`: weight shape `(out_channels, kernel_size, in_channels // groups)`.

In `sanitize`:
```python
if "conv_qk" in k and k.endswith(".weight"):
    out[k] = v.transpose(0, 2, 1)  # (out, in/g, k) -> (out, k, in/g)
```

Bias shape is `(out_channels,)` in both, no transpose needed.

## Stuff to verify when the validation venv is up

- Is `nn.Sequential` available in MLX? If yes, use it for `conv_qk` and `router_mlp`. If not, use a Python list.
- Confirm `nn.Embedding.as_linear` exists at the version of MLX we install.
- Confirm `mlx_lm.load` automatically calls `Model.sanitize` (this is the documented contract; reference: `mlx_lm/utils.py`).

## Helpers from `base.py`

- `BaseModelArgs.from_dict(params)` — load a ModelArgs from a config.json dict.
- `create_attention_mask(h, cache, window_size, return_array)` — standard causal mask creation.
- `scaled_dot_product_attention(queries, keys, values, cache, scale, mask, sinks)` — wraps `mx.fast.scaled_dot_product_attention` with quantized-cache fallback.

We use `scaled_dot_product_attention` from base inside ZayaAttention's forward (Phase 4), not the `mx.fast` primitive directly.

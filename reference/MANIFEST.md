# Reference activations manifest

This index lists every reference dump available for layer-by-layer parity checks against the MLX implementation.

## How to use

Reference activations are produced by `reference/dump_activations.py` and live in `reference/activations/<prompt_id>/`. They are gitignored (large files; reproducible from the script).

To regenerate a single dump (e.g. after a transformers upgrade or hook taxonomy change):

```bash
cd reference
.venv/bin/python dump_activations.py --prompt-id <id>
```

To regenerate all dumps in sequence:

```bash
cd reference
for id in smoke reasoning_short long_context_seed; do
  .venv/bin/python dump_activations.py --prompt-id "$id"
done
```

## Available dumps

| prompt_id | captured_modules | input_shape | captured_at (UTC) | torch | transformers |
|---|---|---|---|---|---|
| smoke | 3046 | (1, 7) | 2026-05-07T00:15:22 | 2.5.1 | 4.57.1 |
| reasoning_short | 3046 | (1, 22) | 2026-05-07T00:15:44 | 2.5.1 | 4.57.1 |
| long_context_seed | 3046 | (1, 78) | 2026-05-07T00:16:04 | 2.5.1 | 4.57.1 |

## Dump sizes on disk

| prompt_id | size |
|---|---|
| smoke | 57 MB |
| reasoning_short | 158 MB |
| long_context_seed | 531 MB |
| **total** | **746 MB** |

## Module key conventions

See `reference/notes/zaya-architecture.md` § "Captured shape inventory" for the canonical list of module keys, their shapes, and dtypes.

## Source provenance

- Model: `Zyphra/ZAYA1-8B` from Hugging Face (BF16 safetensors, 4 shards, ~16 GB)
- Reference framework: `transformers @ git+https://github.com/Zyphra/transformers.git@zaya1`, commit `f0ab5bef`
- Forward dtype: `bfloat16` (saved tensors upcast to fp32 for numpy compatibility)
- Attention implementation: `eager` (the only path we port to MLX)
- KV cache: forward run with `use_cache=False` (no cache state captured)

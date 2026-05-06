# Install log: Zyphra/transformers @ zaya1

**Date:** 2026-05-06
**Platform:** macOS arm64 (M3 Max)
**Python:** 3.11.13 (uv venv)

## Outcome

**SUCCESS on first attempt.** No fallback needed. R2 (Mac install failure) is averted.

## Steps that worked

```bash
cd ~/code/personal/zaya1-mlx/reference
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python \
  "torch>=2.4,<2.6" safetensors "huggingface_hub>=0.26" "numpy>=1.26,<2.0" \
  tqdm sentencepiece "tokenizers>=0.20" protobuf "accelerate>=1.0" einops
uv pip install --python .venv/bin/python \
  "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1"
```

Note: `uv` venvs do not include `pip` by default. Use `uv pip install --python .venv/bin/python ...` instead of `.venv/bin/pip install ...`.

## Versions resolved

- torch: 2.5.1
- transformers: 4.57.1 (Zyphra/transformers @ commit `f0ab5bef9e23b79a9b32f50500d0b52f273a9baf`, branch `zaya1`)
- huggingface_hub: 0.36.2 (downgraded from 1.14.0 by transformers' own deps)
- tokenizers: 0.22.2 (downgraded from 0.23.1)
- numpy: pinned <2.0 per project deps

## Errors encountered

None during install. The only friction was the deprecated `huggingface-cli` (replaced by `hf`) — this was caught in pre-flight and the plan was updated.

## Verification

Both checks passed:

```python
from transformers.models.auto import CONFIG_MAPPING
'zaya' in CONFIG_MAPPING  # True

from transformers.models.zaya.modeling_zaya import ZayaForCausalLM
# imports without error
```

## Source file locations

After install, the model source lives at:
```
.venv/lib/python3.11/site-packages/transformers/models/zaya/
├── __init__.py           (28 lines)
├── configuration_zaya.py (126 lines)
├── modeling_zaya.py      (2,069 lines — likely auto-generated from modular)
└── modular_zaya.py       (2,316 lines — likely the authored source)
```

`modular_zaya.py` is read first in Task 5; `modeling_zaya.py` for any deltas.

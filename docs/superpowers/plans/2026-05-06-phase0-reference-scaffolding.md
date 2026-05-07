# Phase 0: Reference Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a PyTorch reference environment that loads Zyphra/ZAYA1-8B and dumps activation tensors (.npy files) from every major submodule of every layer, on a fixed reference prompt — the foundation for layer-by-layer numerical-parity validation in later phases.

**Architecture:** Isolated `uv` venv inside `~/code/personal/zaya1-mlx/reference/`. PyTorch loads the model once, registers forward hooks on every named submodule of every layer, runs a 32-token forward pass on a fixed prompt, and writes outputs to `reference/activations/<prompt_hash>/L{i}_{module}.npy`. A `MANIFEST.md` records prompt, hash, capture date, torch version, transformers commit. The reference venv is intentionally separate from any MLX environment — they never run in the same process.

**Tech Stack:** Python 3.11, `uv` (venv + dep manager), `torch` (CPU on macOS or CUDA on fallback Linux box), `transformers @ Zyphra/transformers#zaya1`, `huggingface_hub`, `safetensors`, `numpy`.

---

## Pre-flight Setup

Before Task 1, confirm working directory and tooling.

- [ ] **Check `uv` is installed**

Run: `uv --version`
Expected: prints a version (e.g. `uv 0.5.x`). If "command not found", install: `brew install uv`.

- [ ] **Check `hf` CLI is available and authenticated**

Run: `hf auth whoami`
Expected: prints `user=<username>`. The legacy `huggingface-cli` is deprecated; use `hf`.
If not logged in: `hf auth login` and paste a token from https://huggingface.co/settings/tokens (read scope is enough for public models).

- [ ] **Check available disk space**

Run: `df -h ~/.cache/huggingface/hub 2>/dev/null || df -h ~`
Expected: at least 25 GB free. The ZAYA1-8B weights are ~17.7 GB plus PyTorch's BF16 tensors will be loaded into RAM.

---

### Task 1: Create Reference Directory Skeleton

**Files:**
- Create: `~/code/personal/zaya1-mlx/reference/.gitkeep`
- Create: `~/code/personal/zaya1-mlx/reference/activations/.gitkeep`
- Create: `~/code/personal/zaya1-mlx/reference/notes/.gitkeep`
- Create: `~/code/personal/zaya1-mlx/validation/.gitkeep`
- Create: `~/code/personal/zaya1-mlx/scripts/.gitkeep`
- Create: `~/code/personal/zaya1-mlx/zaya1_mlx/__init__.py`

- [ ] **Step 1: Create directory structure**

Run:
```bash
cd ~/code/personal/zaya1-mlx
mkdir -p reference/activations reference/notes validation scripts zaya1_mlx
# .gitkeep only in dirs not already gitignored. activations/ is ignored, skip it.
touch reference/notes/.gitkeep validation/.gitkeep scripts/.gitkeep
```

- [ ] **Step 2: Create empty `zaya1_mlx` package init**

Write to `zaya1_mlx/__init__.py`:
```python
"""zaya1_mlx — thin wrapper exposing the MLX port of ZAYA1-8B from the mlx-lm fork.

Currently a placeholder. Phase 1 will add re-exports.
"""
__version__ = "0.0.0"
```

- [ ] **Step 3: Verify structure**

Run: `find ~/code/personal/zaya1-mlx -type d -not -path '*/.git*' | sort`
Expected output includes: `reference`, `reference/activations`, `reference/notes`, `validation`, `scripts`, `zaya1_mlx`.

- [ ] **Step 4: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/ validation/ scripts/ zaya1_mlx/
git commit -m "Phase 0 task 1: scaffold reference, validation, scripts, zaya1_mlx dirs"
```

---

### Task 2: Bootstrap the Reference Python Environment

**Files:**
- Create: `~/code/personal/zaya1-mlx/reference/pyproject.toml`
- Create: `~/code/personal/zaya1-mlx/reference/.python-version`

- [ ] **Step 1: Write `reference/pyproject.toml`**

```toml
[project]
name = "zaya1-mlx-reference"
version = "0.0.0"
description = "PyTorch reference environment for the ZAYA1-8B MLX port. Runs forward passes and dumps activation tensors for layer-by-layer comparison."
requires-python = ">=3.11,<3.13"

dependencies = [
  # Pinned to versions Zyphra's fork is known to work with as of 2026-05-06.
  # If the install in Task 3 fails, the failure mode plus resolution belong in reference/notes/install-log.md.
  "torch>=2.4,<2.6",
  "safetensors>=0.4",
  "huggingface_hub>=0.26",
  "numpy>=1.26,<2.0",
  "tqdm",
  "sentencepiece",
  "tokenizers>=0.20",
  "protobuf",
  "accelerate>=1.0",
  "einops",
]

[tool.uv]
# Zyphra's transformers fork is installed via Task 3, not as a dep here, because
# pyproject.toml + uv have flaky behavior around git-pinned packages on macOS.
```

- [ ] **Step 2: Pin Python version**

Write to `reference/.python-version`:
```
3.11
```

- [ ] **Step 3: Create the venv**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
uv venv .venv --python 3.11
```
Expected: prints `Using CPython 3.11.x` and `Activate with: source .venv/bin/activate`.

- [ ] **Step 4: Install base deps**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
uv pip install --python .venv/bin/python \
  "torch>=2.4,<2.6" safetensors "huggingface_hub>=0.26" "numpy>=1.26,<2.0" \
  tqdm sentencepiece "tokenizers>=0.20" protobuf "accelerate>=1.0" einops
```
Expected: install completes without error. If torch wheel fails to find macOS arm64 build, fall back to `uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu`.

- [ ] **Step 5: Smoke-test torch on M3 Max**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'mps_available', torch.backends.mps.is_available(), 'cuda_available', torch.cuda.is_available())"
```
Expected: prints torch version, `mps_available True`, `cuda_available False`. We will NOT use MPS (Zyphra's custom kernels likely don't support it); CPU is what we need for reference dumps.

- [ ] **Step 6: Commit**

```bash
cd ~/code/personal/zaya1-mlx
echo ".venv/" > reference/.gitignore
git add reference/pyproject.toml reference/.python-version reference/.gitignore
git commit -m "Phase 0 task 2: bootstrap reference uv venv with torch + base deps"
```

---

### Task 3: Install Zyphra's transformers Fork (with Failure-Mode Documentation)

**Files:**
- Create: `~/code/personal/zaya1-mlx/reference/notes/install-log.md`

This task is high-risk: Zyphra's fork may have CUDA-only or triton-only paths that fail on macOS. Document everything.

- [ ] **Step 1: Attempt direct install of Zyphra's fork**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1" 2>&1 | tee notes/install-log-attempt-1.txt
```
Expected behaviors and what they mean:
- **Success** → proceed to Step 3.
- **Build failure due to triton/CUDA** → proceed to Step 2 (CPU-only fallback).
- **Auth/git issue** → check internet; the URL is public so no token needed.

- [ ] **Step 2 (only if Step 1 failed): Install with no-build-isolation and skip optional CUDA bits**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/pip install --no-build-isolation "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1" 2>&1 | tee notes/install-log-attempt-2.txt
```

If this also fails, the install error is the truth. Read it, then write `reference/notes/install-log.md` documenting:
- What was tried
- Exact error (include the last 30 lines of the failed install)
- Decision: proceed with macOS or fall back to a Linux GPU instance for reference dumps

If falling back to Linux: stop this plan and open a new sub-plan for "remote reference dumps" — out of scope for the current pass.

- [ ] **Step 3: Verify the `zaya` model type is registered**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -c "from transformers.models.auto import CONFIG_MAPPING; print('zaya' in CONFIG_MAPPING)"
```
Expected: `True`. If `False`, the install partially failed; check that `transformers/models/zaya/` was actually added to the installed package.

- [ ] **Step 4: Verify the modeling file is importable**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -c "from transformers.models.zaya.modeling_zaya import ZayaForCausalLM; print(ZayaForCausalLM)"
```
Expected: prints `<class 'transformers.models.zaya.modeling_zaya.ZayaForCausalLM'>`.

- [ ] **Step 5: Write the install log**

Write `reference/notes/install-log.md` with sections:
```markdown
# Install log: Zyphra/transformers @ zaya1

**Date:** 2026-05-06
**Platform:** macOS arm64 (M3 Max)
**Python:** 3.11.x (uv venv)

## Outcome
[SUCCESS or FAILURE]

## Steps that worked
[list the exact pip commands that succeeded]

## Errors encountered (if any)
[paste error excerpts, identify root cause]

## Resolution
[what made it work, or — if it didn't — the fallback decision]
```

- [ ] **Step 6: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/notes/install-log.md
git commit -m "Phase 0 task 3: install Zyphra transformers fork + log install steps"
```

---

### Task 4: Download ZAYA1-8B Weights

**Files:**
- Create: `~/code/personal/zaya1-mlx/scripts/download_weights.sh`

- [ ] **Step 1: Write download script**

Write `scripts/download_weights.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

# Downloads Zyphra/ZAYA1-8B weights to the local Hugging Face cache.
# Re-runnable: HF cache dedupes by content hash so reruns are fast.

REPO="Zyphra/ZAYA1-8B"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not on PATH. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

echo "Downloading $REPO to local HF cache..."
hf download "$REPO" \
  --include "*.safetensors" "*.json" "*.txt" "tokenizer*" \
  --exclude "*.bin" "*.gguf"

echo
echo "Cache location:"
hf cache scan | grep "$REPO" || true
```

Make executable: `chmod +x scripts/download_weights.sh`.

- [ ] **Step 2: Run download**

Run: `~/code/personal/zaya1-mlx/scripts/download_weights.sh`
Expected: 4 safetensors shards (≈17.7 GB total) plus config.json, tokenizer files, special_tokens_map.json. Final output of `huggingface-cli scan-cache` shows the repo.

- [ ] **Step 3: Verify load via `from_pretrained` (config-only, no weights into RAM)**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('Zyphra/ZAYA1-8B')
print('model_type:', cfg.model_type)
print('hidden_size:', cfg.hidden_size)
print('num_hidden_layers:', cfg.num_hidden_layers)
print('num_experts:', getattr(cfg, 'num_experts', None))
print('partial_rotary_factor:', getattr(cfg, 'partial_rotary_factor', None))
print('zaya_use_mod:', getattr(cfg, 'zaya_use_mod', None))
print('zaya_use_eda:', getattr(cfg, 'zaya_use_eda', None))
"
```
Expected: matches the values in design §5 (model_type=zaya, hidden_size=2048, num_hidden_layers=80, num_experts=16, partial_rotary_factor=0.5, zaya_use_mod=True, zaya_use_eda=True).

- [ ] **Step 4: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add scripts/download_weights.sh
git commit -m "Phase 0 task 4: weights download script + config sanity check"
```

---

### Task 5: Read modeling_zaya.py and Document Module Structure

This is a research task. Output is a markdown document that catalogs every class and function we will need to port, plus answers to the open architectural questions from design §5.

**Files:**
- Create: `~/code/personal/zaya1-mlx/reference/notes/zaya-architecture.md`

- [ ] **Step 1: Locate the installed modeling files**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
ZAYA_DIR=$(.venv/bin/python -c "import os, transformers; print(os.path.join(os.path.dirname(transformers.__file__), 'models', 'zaya'))")
echo "Zaya source files at: $ZAYA_DIR"
ls "$ZAYA_DIR"
```
Expected: lists `__init__.py`, `configuration_zaya.py`, `modeling_zaya.py`, `modular_zaya.py`.

- [ ] **Step 2: Read `configuration_zaya.py` end-to-end**

Open the file in your editor. Note in `reference/notes/zaya-architecture.md` (create if absent) every config field that is **not** in design §5's list. Especially watch for: anything with prefix `zaya_`, anything controlling layer composition, anything controlling SSM dimensions.

- [ ] **Step 3: Read `modeling_zaya.py` end-to-end and catalog classes**

Output goes to `reference/notes/zaya-architecture.md`. Expected sections:

```markdown
# ZAYA1 Architecture Notes

**Source:** Zyphra/transformers @ zaya1, `transformers/models/zaya/modeling_zaya.py`
**Read date:** 2026-05-06
**File length:** [LOC]

## Class catalog

For each class found, in order of appearance in the file, document:

### ClassName

- **Purpose:** [one sentence]
- **Inputs:** [tensor shapes / dtypes / required kwargs]
- **Outputs:** [tensor shapes / dtypes]
- **Dependencies:** [which other classes it instantiates]
- **MLX porting notes:** [anything that needs special handling — custom kernel, fp32 path, dtype boundary]

## Open question answers

### Q1: Layer schedule — interleaved or hybrid?
[answer with file:line citation]

### Q2: MoE coverage — every layer or subset?
[answer with file:line citation]

### Q3: ZayaRMSNorm semantics — different from stock?
[answer with file:line citation]

### Q4: scale_residual_merge formula
[answer with file:line citation]

### Q5: EDA exact form
[answer with file:line citation]

## Forward-pass call graph

[tree of which class's forward calls which, for the decoder layer specifically]

## Submodules to hook in dump_activations.py

[ordered list — this is what Task 6 consumes]
```

- [ ] **Step 4: Read `modular_zaya.py` for any deltas**

Note any subclassing or overrides that change behavior. Add to the architecture doc.

- [ ] **Step 5: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/notes/zaya-architecture.md
git commit -m "Phase 0 task 5: catalog ZAYA1 architecture from PyTorch source"
```

---

### Task 6: Choose Reference Prompts and Hook Taxonomy

**Files:**
- Modify: `~/code/personal/zaya1-mlx/reference/notes/zaya-architecture.md` (append "Hook taxonomy" section)
- Create: `~/code/personal/zaya1-mlx/reference/prompts.json`

- [ ] **Step 1: Pick reference prompts**

Write `reference/prompts.json`:
```json
{
  "prompts": [
    {
      "id": "smoke",
      "text": "The capital of France is",
      "max_new_tokens": 4,
      "purpose": "smoke test: short, deterministic, easy to eyeball"
    },
    {
      "id": "reasoning_short",
      "text": "<think>If x + 3 = 7, what is x? Solve step by step.</think>",
      "max_new_tokens": 16,
      "purpose": "exercises the Qwen3-style reasoning token path"
    },
    {
      "id": "long_context_seed",
      "text": "Once upon a time, in a kingdom far away, there lived a young scholar who dreamed of mastering the art of state-space models. Every day, she would walk to the library and read about recurrence, scans, and the dance between hidden states.",
      "max_new_tokens": 1,
      "purpose": "32+ token sequence to exercise position-dependent paths (RoPE, KV cache); single output token to keep dumps small"
    }
  ]
}
```

- [ ] **Step 2: Hook taxonomy is in the architecture doc**

The original speculative taxonomy in this plan was based on a wrong architecture model (Mamba SSM, every-layer MoE). After Task 5's source read, the **canonical hook taxonomy is now in `reference/notes/zaya-architecture.md` § "Submodules to hook in dump_activations.py"** and reflects:

- Strict ATT/MoE alternation (no SSM).
- Different hook sets for even (ATT) vs odd (MoE) layers.
- CCA-specific submodule keys (linear_q, linear_k, val_proj1, val_proj2, conv_qk[0], conv_qk[1]) instead of SSM keys.
- Router-internal keys (down_proj, rmsnorm_eda, router_mlp[0/2/4]).
- Per-expert keys (only the chosen expert produces output for any given token under top-1 routing).

Implementation in Task 7 walks every named submodule and dumps its output, so the taxonomy is more documentation than configuration — the dump will produce one .npy per submodule whether it appears in the table or not. The table tells future readers what each key means.

- [ ] **Step 3: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/prompts.json reference/notes/zaya-architecture.md
git commit -m "Phase 0 task 6: define reference prompts and hook taxonomy"
```

---

### Task 7: Implement `dump_activations.py`

**Files:**
- Create: `~/code/personal/zaya1-mlx/reference/dump_activations.py`
- Create: `~/code/personal/zaya1-mlx/reference/test_dump_activations.py`

This task IS TDD-shaped: the contract is "given a prompt id and the model, produce one .npy file per (layer, module_key) pair, plus a manifest entry." We test the file-output contract.

- [ ] **Step 1: Write the failing test**

Write `reference/test_dump_activations.py`:
```python
"""Test the dump_activations contract.

We test the FILE OUTPUT contract, not the model itself: did the dump produce
the expected file structure for a known prompt?
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REFERENCE_DIR = Path(__file__).parent
SCRIPT = REFERENCE_DIR / "dump_activations.py"
ACTIVATIONS_ROOT = REFERENCE_DIR / "activations"


def _run_dump(prompt_id: str = "smoke") -> Path:
    """Run dump_activations.py and return the output directory."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--prompt-id", prompt_id, "--max-layers", "2"],
        cwd=REFERENCE_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(f"dump_activations.py exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    # Output dir is printed on the last non-empty stdout line.
    out_path = Path([line for line in result.stdout.strip().splitlines() if line][-1])
    assert out_path.exists(), f"Reported output dir does not exist: {out_path}"
    return out_path


def test_dump_produces_manifest():
    out_dir = _run_dump("smoke")
    manifest = out_dir / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["prompt_id"] == "smoke"
    assert "torch_version" in data
    assert "transformers_commit" in data
    assert "captured_modules" in data
    assert isinstance(data["captured_modules"], list)
    assert len(data["captured_modules"]) > 0


def test_dump_produces_npy_files_for_layer_zero():
    out_dir = _run_dump("smoke")
    npys = list(out_dir.glob("L0_*.npy"))
    assert len(npys) > 0, f"No L0_* npy files in {out_dir}"
    # Smoke check: each file is loadable as numpy
    for path in npys:
        arr = np.load(path)
        assert arr.size > 0, f"Empty array in {path}"


def test_dump_layer_count_respects_flag():
    out_dir = _run_dump("smoke")
    layer_indices = sorted({int(p.name.split("_")[0][1:]) for p in out_dir.glob("L*_*.npy")})
    assert layer_indices == [0, 1], f"Expected layers [0, 1] with --max-layers 2, got {layer_indices}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -m pytest test_dump_activations.py -v
```
Expected: tests fail because `dump_activations.py` does not exist yet (or exits non-zero). The `dump_activations.py exited` failure message is fine — that's the failing-test signal.

- [ ] **Step 3: Implement `dump_activations.py`**

Write `reference/dump_activations.py`:
```python
"""Run a forward pass on Zyphra/ZAYA1-8B and dump every layer's submodule outputs to .npy.

Usage:
    python dump_activations.py --prompt-id smoke
    python dump_activations.py --prompt-id smoke --max-layers 2  # debugging

Outputs:
    activations/<prompt_id>/L{i}_{module_key}.npy
    activations/<prompt_id>/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ID = "Zyphra/ZAYA1-8B"
SCRIPT_DIR = Path(__file__).parent
PROMPTS_FILE = SCRIPT_DIR / "prompts.json"
ACTIVATIONS_ROOT = SCRIPT_DIR / "activations"


def load_prompts() -> Dict[str, dict]:
    data = json.loads(PROMPTS_FILE.read_text())
    return {p["id"]: p for p in data["prompts"]}


def hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def get_transformers_commit() -> str:
    """Best-effort: return the installed transformers package commit/version string."""
    try:
        import transformers
        return getattr(transformers, "__version__", "unknown")
    except Exception:
        return "unknown"


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach and convert a tensor to numpy, preserving dtype where possible."""
    t = t.detach().cpu()
    # numpy has no native bf16; upcast to float32 when saving so .npy is loadable elsewhere.
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return t.numpy()


def register_hooks(model, captured: Dict[str, np.ndarray], max_layers: int | None) -> List[torch.utils.hooks.RemovableHandle]:
    """Walk the model and register a forward hook on every named submodule.

    Hook saves output tensors keyed by 'L{layer_idx}_{leaf_module_path}' or 'global_{module_path}'
    if the submodule is outside a numbered layer.

    A leaf module is one whose immediate forward output is a tensor or simple tuple of tensors.
    """
    handles = []

    # Resolve the layer container. For most HF models this is `model.model.layers`.
    # ZAYA1 may use a different name; we look for the first attribute that is a ModuleList of length == num_hidden_layers.
    try:
        layers = model.model.layers
    except AttributeError:
        # Fallback: search for a ModuleList of expected size
        cfg = model.config
        target_n = cfg.num_hidden_layers
        layers = None
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) == target_n:
                layers = mod
                break
        if layers is None:
            raise RuntimeError(f"Could not locate decoder layers ModuleList of length {target_n}")

    def make_hook(key: str):
        def hook(_module, _inputs, output):
            # Output may be a tensor, a tuple, or a dict-like structure. Save first tensor we find.
            if isinstance(output, torch.Tensor):
                captured[key] = _to_numpy(output)
            elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                captured[key] = _to_numpy(output[0])
            elif hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
                captured[key] = _to_numpy(output.last_hidden_state)
            # else: skip non-tensor outputs silently
        return hook

    for layer_idx, layer in enumerate(layers):
        if max_layers is not None and layer_idx >= max_layers:
            break
        for sub_name, sub_module in layer.named_modules():
            if sub_name == "":
                # The layer itself
                key = f"L{layer_idx}_layer_out"
            else:
                key = f"L{layer_idx}_{sub_name.replace('.', '_')}_out"
            handle = sub_module.register_forward_hook(make_hook(key))
            handles.append(handle)

    return handles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-layers", type=int, default=None, help="Optional: only hook the first N layers (debugging)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    prompts = load_prompts()
    if args.prompt_id not in prompts:
        print(f"Unknown prompt id: {args.prompt_id}. Available: {list(prompts)}", file=sys.stderr)
        return 2
    prompt = prompts[args.prompt_id]

    out_dir = ACTIVATIONS_ROOT / args.prompt_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {REPO_ID} (this is slow on first run)...", flush=True)
    cfg = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    captured: Dict[str, np.ndarray] = {}
    handles = register_hooks(model, captured, args.max_layers)

    inputs = tokenizer(prompt["text"], return_tensors="pt")

    print(f"Forward pass: prompt_id={args.prompt_id}, len(input_ids)={inputs.input_ids.shape}", flush=True)
    with torch.no_grad():
        _ = model(**inputs)

    for handle in handles:
        handle.remove()

    print(f"Captured {len(captured)} tensors", flush=True)
    for key, arr in captured.items():
        np.save(out_dir / f"{key}.npy", arr)

    manifest = {
        "prompt_id": args.prompt_id,
        "prompt_text": prompt["text"],
        "prompt_hash": hash_prompt(prompt["text"]),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "transformers_commit": get_transformers_commit(),
        "dtype": args.dtype,
        "max_layers": args.max_layers,
        "input_ids_shape": list(inputs.input_ids.shape),
        "captured_modules": sorted(captured.keys()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(out_dir.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -m pytest test_dump_activations.py -v
```
Expected: all 3 tests pass. The first test will be slow (loads the full model once) — subsequent tests reuse via Python's import cache; expect 5–15 minutes total on M3 Max for the first pass.

If `test_dump_layer_count_respects_flag` fails with layers other than `[0, 1]`, the layer-resolution heuristic in `register_hooks` is wrong. Fix it by reading `reference/notes/zaya-architecture.md` for the actual attribute path to the decoder layers, then update `register_hooks` to use the correct path.

- [ ] **Step 5: Add pytest to dev deps and pin**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/pip install pytest
```
And add `pytest>=8.0` to `[project].dependencies` in `pyproject.toml`. (We deliberately add it after first use rather than upfront, so the bootstrap install is minimal.)

- [ ] **Step 6: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/dump_activations.py reference/test_dump_activations.py reference/pyproject.toml
git commit -m "Phase 0 task 7: implement dump_activations.py with file-output contract tests"
```

---

### Task 8: Run Full Reference Dump

**Files:**
- Modify: `~/code/personal/zaya1-mlx/reference/notes/zaya-architecture.md` (append "Captured shape inventory")

- [ ] **Step 1: Run dump for each prompt**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python dump_activations.py --prompt-id smoke
.venv/bin/python dump_activations.py --prompt-id reasoning_short
.venv/bin/python dump_activations.py --prompt-id long_context_seed
```
Expected: each command prints a path under `activations/<prompt_id>/`. Each run takes 5-15 minutes on M3 Max (CPU). RAM usage will spike to ~17 GB during forward pass — watch Activity Monitor and close other heavy apps.

- [ ] **Step 2: Spot-check the output**

Run:
```bash
cd ~/code/personal/zaya1-mlx/reference
ls activations/smoke | head -20
.venv/bin/python -c "
import json, numpy as np
from pathlib import Path
m = json.loads(Path('activations/smoke/manifest.json').read_text())
print('captured_modules count:', len(m['captured_modules']))
print('first 5:', m['captured_modules'][:5])
arr = np.load('activations/smoke/L0_layer_out.npy')
print('L0_layer_out shape:', arr.shape, 'dtype:', arr.dtype)
"
```
Expected: count is large (hundreds; one per submodule × 80 layers); shape of `L0_layer_out` is `(1, seq_len, 2048)`; dtype is `float32` (saved up-cast from bf16).

- [ ] **Step 3: Append "Captured shape inventory" to architecture doc**

In `reference/notes/zaya-architecture.md`, append:
```markdown
## Captured shape inventory (smoke prompt)

For each unique module suffix (everything after `L{i}_`), record:
- Suffix
- Shape
- Dtype

This becomes the contract that the MLX implementation must match in Phase 2+.

[fill in by reading the manifest + a sample of .npy files]
```

Generate the inventory programmatically:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -c "
import json, numpy as np
from pathlib import Path
from collections import defaultdict

acts = Path('activations/smoke')
m = json.loads((acts / 'manifest.json').read_text())
suffixes = defaultdict(list)
for name in m['captured_modules']:
    # 'L7_self_attn_q_proj_out' -> suffix 'self_attn_q_proj_out', layer 7
    parts = name.split('_', 1)
    if not (parts[0].startswith('L') and parts[0][1:].isdigit()):
        continue
    layer = int(parts[0][1:])
    suffix = parts[1]
    suffixes[suffix].append((layer, name))

print('| suffix | shape | dtype |')
print('|---|---|---|')
for suffix, occurrences in sorted(suffixes.items()):
    layer, name = occurrences[0]
    arr = np.load(acts / f'{name}.npy')
    print(f'| {suffix} | {tuple(arr.shape)} | {arr.dtype} |')
" >> notes/_inventory.tmp.md
cat notes/_inventory.tmp.md
```
Then paste into the architecture doc and remove `notes/_inventory.tmp.md`.

- [ ] **Step 4: Commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/notes/zaya-architecture.md
git commit -m "Phase 0 task 8: capture full reference activations + shape inventory"
```

---

### Task 9: Write MANIFEST.md and Update STATUS

**Files:**
- Create: `~/code/personal/zaya1-mlx/reference/MANIFEST.md`
- Modify: `~/code/personal/zaya1-mlx/STATUS.md`

- [ ] **Step 1: Write `reference/MANIFEST.md`**

```markdown
# Reference activations manifest

This index lists every reference dump available for layer-by-layer parity checks against the MLX implementation.

## How to use

Reference activations are produced by `reference/dump_activations.py`. They live in `reference/activations/<prompt_id>/` and are gitignored (large files; reproducible from the script).

To regenerate (e.g. after a transformers update):
```
cd reference
.venv/bin/python dump_activations.py --prompt-id <id>
```

## Available dumps

| prompt_id | text | input_len | captured_at | torch | transformers |
|---|---|---|---|---|---|
| smoke | "The capital of France is" | [from manifest] | [date] | [version] | [version] |
| reasoning_short | "<think>If x + 3 = 7, what is x? Solve step by step.</think>" | [from manifest] | [date] | [version] | [version] |
| long_context_seed | "Once upon a time..." (≥32 tokens) | [from manifest] | [date] | [version] | [version] |

[Auto-generate this table from the per-prompt manifest.json files. Update whenever a new prompt is added or dumps are refreshed.]

## Module key conventions

See `reference/notes/zaya-architecture.md` § "Captured shape inventory" for the canonical list of module keys and their shapes/dtypes.
```

Auto-fill the table:
```bash
cd ~/code/personal/zaya1-mlx/reference
.venv/bin/python -c "
import json
from pathlib import Path
for d in sorted(Path('activations').iterdir()):
    mf = d / 'manifest.json'
    if not mf.exists(): continue
    m = json.loads(mf.read_text())
    print(f\"{m['prompt_id']} | {m['prompt_text'][:60]} | {m['input_ids_shape']} | {m['captured_at']} | {m['torch_version']} | {m['transformers_commit']}\")
"
```
Paste into the table, formatted as markdown.

- [ ] **Step 2: Update `STATUS.md`**

Replace its content with:
```markdown
# Status

**Last updated:** [date]

## Current phase

Phase 1 — Skeleton + weight loading (not yet started).

## What's done

- Phase 0 complete:
  - Reference uv venv set up with PyTorch + Zyphra transformers fork
  - ZAYA1-8B weights downloaded
  - `modeling_zaya.py` read and architecture documented at `reference/notes/zaya-architecture.md`
  - `dump_activations.py` implemented + tested (file-output contract)
  - Reference activations captured for 3 prompts (smoke, reasoning_short, long_context_seed)
  - `reference/MANIFEST.md` indexes available dumps

## What's next

Phase 1: write Plan 2 (skeleton + weight loading in mlx-lm fork). Plan 2 will use the architecture notes from Phase 0 to design the `ModelArgs` dataclass and `sanitize(weights)` weight-key remapping.

## Blockers

None.

## Reference activation paths

See `reference/MANIFEST.md` and `reference/notes/zaya-architecture.md`.
```

- [ ] **Step 3: Final commit**

```bash
cd ~/code/personal/zaya1-mlx
git add reference/MANIFEST.md STATUS.md
git commit -m "Phase 0 complete: reference manifest + status update"
git push origin main
```

---

## Phase 0 Gate Verification

Before declaring Phase 0 done, the following must be true. Verify each:

- [ ] `reference/.venv/bin/python -c "import transformers; from transformers.models.zaya.modeling_zaya import ZayaForCausalLM"` succeeds.
- [ ] `hf cache scan | grep ZAYA1-8B` shows the weights are downloaded.
- [ ] `reference/notes/zaya-architecture.md` exists, contains a class catalog, and answers all 5 open questions from design §5.
- [ ] `reference/activations/smoke/manifest.json` exists; `captured_modules` has length > 100.
- [ ] `reference/activations/smoke/L0_*.npy` files exist and are loadable as numpy arrays.
- [ ] `reference/test_dump_activations.py` passes.
- [ ] `STATUS.md` reflects Phase 0 complete.
- [ ] All commits pushed to `https://github.com/zappleg8/zaya1-mlx`.

If any of these fail, do not write Plan 2 yet — fix the failure, then revalidate.

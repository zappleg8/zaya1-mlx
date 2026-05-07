"""Run a forward pass on Zyphra/ZAYA1-8B and dump every named submodule's output to .npy.

Usage:
    python dump_activations.py --prompt-id smoke
    python dump_activations.py --prompt-id smoke --max-layers 2  # debugging

Outputs:
    activations/<prompt_id>/L{i}_<submodule_path>_out.npy   (per-layer hooks)
    activations/<prompt_id>/global_<submodule_path>_out.npy  (model-level hooks)
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
import torch.nn as nn
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
    try:
        import transformers
        version = getattr(transformers, "__version__", "unknown")
        try:
            import transformers.models.zaya
            zaya_path = Path(transformers.models.zaya.__file__).parent
            return f"{version} (zaya at {zaya_path})"
        except Exception:
            return version
    except Exception:
        return "unknown"


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach a tensor and convert to numpy. bf16 is upcast to fp32 (numpy has no native bf16)."""
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return t.numpy()


def _save_output(captured: Dict[str, np.ndarray], key_base: str, output) -> None:
    """Save whatever a forward hook saw — handle tensor, tuple, or ModelOutput."""
    if isinstance(output, torch.Tensor):
        captured[f"{key_base}_out"] = _to_numpy(output)
    elif isinstance(output, (tuple, list)):
        tensors = [x for x in output if isinstance(x, torch.Tensor)]
        # 3-tuple of same-dtype tensors → CCA-style (Q, K, V)
        if len(tensors) == 3 and tensors[0].dtype == tensors[1].dtype == tensors[2].dtype:
            captured[f"{key_base}_q"] = _to_numpy(tensors[0])
            captured[f"{key_base}_k"] = _to_numpy(tensors[1])
            captured[f"{key_base}_v"] = _to_numpy(tensors[2])
        # 2-tuple of same-dtype tensors → e.g. rotary embedding (cos, sin)
        elif len(tensors) == 2 and tensors[0].dtype == tensors[1].dtype:
            captured[f"{key_base}_0"] = _to_numpy(tensors[0])
            captured[f"{key_base}_1"] = _to_numpy(tensors[1])
        elif len(tensors) > 0:
            captured[f"{key_base}_out"] = _to_numpy(tensors[0])
    elif hasattr(output, "last_hidden_state") and isinstance(
        output.last_hidden_state, torch.Tensor
    ):
        captured[f"{key_base}_out"] = _to_numpy(output.last_hidden_state)
    elif hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
        captured[f"{key_base}_logits"] = _to_numpy(output.logits)
    # else: skip non-tensor outputs


def register_hooks(
    model: nn.Module,
    captured: Dict[str, np.ndarray],
    max_layers: int | None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks on every named submodule.

    Layer-internal modules are keyed `L{idx}_{path}`.
    Modules outside the layer list are keyed `global_{path}`.

    With max_layers set, only hooks layers [0, max_layers) — useful for fast tests.
    """
    handles = []

    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None or not isinstance(layers, nn.ModuleList):
        target_n = model.config.num_hidden_layers
        for _, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) == target_n:
                layers = mod
                break
    if layers is None:
        raise RuntimeError(
            f"Could not locate decoder layers ModuleList of length "
            f"{model.config.num_hidden_layers}"
        )

    layer_id_set = {id(layer) for layer in layers}
    layer_idx_by_id = {id(layer): i for i, layer in enumerate(layers)}

    def make_layer_hook(key: str):
        def hook(_module, _inputs, output):
            _save_output(captured, key, output)
        return hook

    def make_global_hook(key: str):
        def hook(_module, _inputs, output):
            _save_output(captured, key, output)
        return hook

    for layer_idx, layer in enumerate(layers):
        if max_layers is not None and layer_idx >= max_layers:
            break
        for sub_name, sub_module in layer.named_modules():
            key = (
                f"L{layer_idx}_layer"
                if sub_name == ""
                else f"L{layer_idx}_{sub_name.replace('.', '_')}"
            )
            handles.append(sub_module.register_forward_hook(make_layer_hook(key)))

    for full_name, mod in model.named_modules():
        if full_name == "":
            continue
        if id(mod) in layer_id_set:
            continue
        is_under_layer = False
        for layer_id, layer_idx in layer_idx_by_id.items():
            for layer in layers:
                if id(layer) == layer_id:
                    for sub_name, sub_mod in layer.named_modules():
                        if id(sub_mod) == id(mod):
                            is_under_layer = True
                            break
                    if is_under_layer:
                        break
            if is_under_layer:
                break
        if is_under_layer:
            continue
        key = f"global_{full_name.replace('.', '_')}"
        handles.append(mod.register_forward_hook(make_global_hook(key)))

    return handles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Only hook the first N layers (for fast iteration during testing)",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    prompts = load_prompts()
    if args.prompt_id not in prompts:
        print(
            f"Unknown prompt id: {args.prompt_id}. Available: {list(prompts)}",
            file=sys.stderr,
        )
        return 2
    prompt = prompts[args.prompt_id]

    out_dir = ACTIVATIONS_ROOT / args.prompt_id
    if out_dir.exists():
        for f in out_dir.iterdir():
            if f.is_file():
                f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {REPO_ID} (slow on first run; bf16 weights are ~16GB)...", flush=True)
    cfg = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=False)
    cfg._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        config=cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    captured: Dict[str, np.ndarray] = {}
    handles = register_hooks(model, captured, args.max_layers)
    print(f"Registered {len(handles)} hooks", flush=True)

    inputs = tokenizer(prompt["text"], return_tensors="pt")
    print(
        f"Forward pass: prompt_id={args.prompt_id}, input_ids shape={tuple(inputs.input_ids.shape)}",
        flush=True,
    )

    with torch.no_grad():
        _ = model(**inputs, use_cache=False)

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

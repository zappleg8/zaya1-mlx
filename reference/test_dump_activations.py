"""Test the dump_activations contract.

We test the FILE OUTPUT contract, not the model itself. The dump is run ONCE
per test session via a session-scoped fixture (model load is ~5-15 min on
M3 Max CPU) and all assertions read its output.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REFERENCE_DIR = Path(__file__).parent
SCRIPT = REFERENCE_DIR / "dump_activations.py"


@pytest.fixture(scope="session")
def smoke_dump_dir() -> Path:
    """Run dump_activations once with --max-layers 2 and return the output dir."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prompt-id",
            "smoke",
            "--max-layers",
            "2",
        ],
        cwd=REFERENCE_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        pytest.fail(
            f"dump_activations.py exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    out_path = Path(
        [line for line in result.stdout.strip().splitlines() if line][-1]
    )
    assert out_path.exists(), f"Reported output dir does not exist: {out_path}"
    return out_path


def test_dump_produces_manifest(smoke_dump_dir: Path):
    manifest_file = smoke_dump_dir / "manifest.json"
    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text())
    assert data["prompt_id"] == "smoke"
    assert "torch_version" in data
    assert "transformers_commit" in data
    assert "captured_modules" in data
    assert isinstance(data["captured_modules"], list)
    assert len(data["captured_modules"]) > 0


def test_dump_produces_npy_files_for_layer_zero(smoke_dump_dir: Path):
    npys = list(smoke_dump_dir.glob("L0_*.npy"))
    assert len(npys) > 0, f"No L0_* npy files in {smoke_dump_dir}"
    for path in npys:
        arr = np.load(path)
        assert arr.size > 0, f"Empty array in {path}"


def test_dump_layer_count_respects_flag(smoke_dump_dir: Path):
    layer_indices = sorted(
        {
            int(p.name.split("_")[0][1:])
            for p in smoke_dump_dir.glob("L*_*.npy")
            if p.name.split("_")[0][1:].isdigit()
        }
    )
    assert layer_indices == [0, 1], (
        f"Expected layers [0, 1] with --max-layers 2, got {layer_indices}"
    )


def test_dump_captures_layer0_attention_submodules(smoke_dump_dir: Path):
    """Layer 0 is an ATT layer (CCA attention). Verify CCA-specific submodules dumped."""
    npy_names = {p.stem for p in smoke_dump_dir.glob("L0_*.npy")}
    expected_substrings = ["self_attn_qkv_linear_q", "self_attn_qkv_linear_k", "self_attn_o_proj"]
    for needle in expected_substrings:
        matches = [n for n in npy_names if needle in n]
        assert matches, f"No L0_* dump contains '{needle}'. Got: {sorted(npy_names)[:30]}"


def test_dump_captures_layer1_moe_submodules(smoke_dump_dir: Path):
    """Layer 1 is a MoE layer. Verify router and expert submodules dumped."""
    npy_names = {p.stem for p in smoke_dump_dir.glob("L1_*.npy")}
    expected_substrings = ["zaya_block_router_down_proj", "zaya_block_experts"]
    for needle in expected_substrings:
        matches = [n for n in npy_names if needle in n]
        assert matches, f"No L1_* dump contains '{needle}'. Got: {sorted(npy_names)[:30]}"

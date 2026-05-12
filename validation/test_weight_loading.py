"""Phase 1 gate test: load every Zaya safetensor weight into the MLX skeleton.

The skeleton's __call__ methods may be stubs; only the parameter shapes need
to match. Test passes when:
  - mlx_lm.load("Zyphra/ZAYA1-8B") completes without error
  - All 2,483 HF tensors map to skeleton params (no missing, no leftovers)
  - tie_word_embeddings is honored (no separate lm_head.weight loaded)
  - Parameter count matches the HF total (~8.84B)
"""
import mlx.core as mx
import mlx.nn as nn
import pytest


EXPECTED_HF_TENSORS = 2483
EXPECTED_PARAMS_LOWER = 8_800_000_000
EXPECTED_PARAMS_UPPER = 8_900_000_000


def _count_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    total = 0
    for _, v in tree_flatten(model.parameters()):
        if isinstance(v, mx.array):
            total += v.size
    return total


def test_model_type_is_zaya(loaded_model):
    assert loaded_model.model_type == "zaya"


def test_layer_count(loaded_model):
    assert len(loaded_model.layers) == 80


def test_layers_alternate_att_moe(loaded_model):
    """Even indices: ATT layers (have self_attn). Odd: MoE (have zaya_block)."""
    for i, layer in enumerate(loaded_model.layers):
        if i % 2 == 0:
            assert hasattr(layer, "self_attn"), f"layer {i} should be ATT (have self_attn)"
            assert not hasattr(layer, "zaya_block"), f"layer {i} should be ATT (no zaya_block)"
        else:
            assert hasattr(layer, "zaya_block"), f"layer {i} should be MoE (have zaya_block)"
            assert not hasattr(layer, "self_attn"), f"layer {i} should be MoE (no self_attn)"


def test_total_param_count(loaded_model):
    n = _count_params(loaded_model)
    assert EXPECTED_PARAMS_LOWER <= n <= EXPECTED_PARAMS_UPPER, (
        f"Expected ~8.84B params, got {n:,}"
    )


def test_embed_and_lm_head_share_weights(loaded_model):
    """tie_word_embeddings: lm_head should be None (Llama style: use as_linear)."""
    assert getattr(loaded_model, "lm_head", None) is None
    embed = loaded_model.model.embed_tokens
    assert embed.weight.shape == (262272, 2048)


def test_layer_0_att_shapes(loaded_model):
    """Spot-check CCA shapes on layer 0."""
    cca = loaded_model.layers[0].self_attn.qkv
    assert cca.linear_q.weight.shape == (1024, 2048)
    assert cca.linear_k.weight.shape == (256, 2048)
    assert cca.val_proj1.weight.shape == (128, 2048)
    assert cca.val_proj2.weight.shape == (128, 2048)
    # MLX nn.Conv1d weight shape: (out_channels, kernel_size, in_channels // groups)
    assert cca.conv_qk.layers[0].weight.shape == (1280, 2, 1)  # depthwise: in/groups=1
    assert cca.conv_qk.layers[1].weight.shape == (1280, 2, 128)  # grouped 10: in/groups=128
    assert cca.temp.shape == (2,)
    o_proj_weight = loaded_model.layers[0].self_attn.o_proj.weight
    assert o_proj_weight.shape == (2048, 1024)


def test_layer_1_moe_shapes_and_no_eda(loaded_model):
    """Spot-check router on layer 1; verify EDA scale absent (zaya_first_layer=1)."""
    router = loaded_model.layers[1].zaya_block.router
    assert router.down_proj.weight.shape == (256, 2048)
    assert router.down_proj.bias.shape == (256,)
    assert router.balancing_biases.shape == (17,)
    assert getattr(router, "router_states_scale", None) is None
    experts = loaded_model.layers[1].zaya_block.experts.local_experts
    assert len(experts) == 16
    assert experts[0].linear_fc1.weight.shape == (4096, 2048)
    assert experts[0].linear_fc2.weight.shape == (2048, 2048)


def test_layer_3_moe_has_eda(loaded_model):
    """Layer 3 is a non-first MoE layer; EDA scale must exist."""
    router = loaded_model.layers[3].zaya_block.router
    assert router.router_states_scale.shape == (256,)

"""Phase 10 gate test: cached generation must produce the same tokens as
the slow recompute-each-step path.

If they diverge, the cache is implementing something incorrectly (most
likely the CCA conv_states or prev_hs handling during single-token
decode).
"""
import mlx.core as mx
import pytest


def _greedy_recompute(model, tokenizer, prompt: str, n_new: int):
    """Reference path: recompute the full sequence at each step."""
    ids = list(tokenizer.encode(prompt))
    for _ in range(n_new):
        inputs = mx.array([ids], dtype=mx.int32)
        logits = model(inputs)
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        ids.append(tok)
        if tok == tokenizer.eos_token_id:
            break
    return ids


def _greedy_cached(model, tokenizer, prompt: str, n_new: int):
    """Fast path: prefill once, then single-token decode with cache."""
    cache = model.make_cache()
    ids = list(tokenizer.encode(prompt))
    # Prefill on the full prompt.
    inputs = mx.array([ids], dtype=mx.int32)
    logits = model(inputs, cache=cache)
    tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    ids.append(tok)
    if tok == tokenizer.eos_token_id:
        return ids
    for _ in range(n_new - 1):
        inputs = mx.array([[tok]], dtype=mx.int32)
        logits = model(inputs, cache=cache)
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        ids.append(tok)
        if tok == tokenizer.eos_token_id:
            break
    return ids


def test_cached_generation_matches_recompute(loaded_model):
    """Generate 8 tokens via both paths; the resulting id sequences must match."""
    from mlx_lm import load
    _, tokenizer = load("Zyphra/ZAYA1-8B")

    prompt = "What is the capital of France?"
    # Use a longer prompt via chat template (model expects it).
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    n_new = 8
    recomputed = _greedy_recompute(loaded_model, tokenizer, chat_prompt, n_new)
    cached = _greedy_cached(loaded_model, tokenizer, chat_prompt, n_new)

    # Pad to same length for comparison.
    min_len = min(len(recomputed), len(cached))
    assert recomputed[:min_len] == cached[:min_len], (
        f"Cached and recompute paths diverged.\n"
        f"  recomputed: {recomputed}\n"
        f"  cached:     {cached}"
    )

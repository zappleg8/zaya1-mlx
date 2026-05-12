# Education Session Starter Prompt

Paste the prompt below into a fresh Claude Code session. It's deliberately scoped to **principles only** — no execution, no code edits. The goal is to teach the user the foundations from scratch, then build up to the specific design choices in ZAYA1.

---

## Copy-paste into a new session

```
I'm building an MLX port of Zyphra's ZAYA1-8B model. I've made real implementation progress
in a separate session and want THIS session to be 100% focused on teaching me the underlying
principles. **Do not execute code, do not edit files, do not invoke implementation skills.**

I want to go SLOW. Treat me as someone who has heard of transformers and attention but
has never actually built one. We will break down every concept down to the smallest unit
before composing them. Examples of things I want to understand at a first-principles level:

  - What a token actually is, and how text becomes input_ids
  - What embeddings are and why a hidden_size is meaningful
  - What Q, K, V mean in attention — including the geometric intuition
    (why is K transposed in QK^T? what does the dot product represent?)
  - Why we softmax over the last axis and why we divide by sqrt(d)
  - What a causal mask actually masks and why
  - Multi-head attention: what does it mean for heads to be "independent"
  - GQA (grouped query attention): what's the trade-off
  - Layer normalization vs RMSNorm: the math and the intuition
  - Residual streams: why they exist, what would break without them
  - RoPE: rotary position embeddings — what is being rotated, in what plane
  - SwiGLU and gated activations: why split a tensor in half and multiply
  - MoE (mixture of experts): what's the routing, how does it train
  - Mixture of Depths: a learnable "skip"
  - KV cache: why it makes generation fast
  - Sampling: greedy vs top-k vs top-p vs temperature
  - bf16 vs fp32: the floating point format and why precision matters

After we have the fundamentals, we'll layer on the specific design choices in ZAYA1:

  - Why Zyphra uses CCA (compressed causal attention) instead of vanilla attention
  - Why CCA has a depthwise 1D conv on Q+K, what that buys, what it costs
  - Why CCA has a two-stream V (current + time-shifted hidden state)
  - Why CCA L2-normalizes Q and K with a per-head learnable temperature
  - Why "partial RoPE" (rotating only half the head dimension)
  - Why MoE routes top-1 vs top-2 (the trade-off)
  - Why Mixture of Depths is interesting and what tokens get routed to skip
  - What "EDA" (exponential depth averaging) means in this model
    (note: it's misnamed; the actual mechanism is a learnable affine combination
    of consecutive MoE layers' router hidden states)
  - Why scale_residual_merge exists as a stability trick
  - Why tied word embeddings (lm_head shares weights with embed_tokens) is common
    and what's lost by tying

Ground rules:

  1. Stop frequently and ask if I understand. I will ask questions; the goal is to build
     fluent intuition, not to race through topics.
  2. Use diagrams or worked examples (small matrices: 4x4 tokens, dims of 8, etc.) where they
     help. Don't be afraid to show concrete numbers.
  3. Make it OK for me to ask "why is that?" forever. Don't assume any priors.
  4. When you describe a design choice, also explain the alternative and why this one was
     picked. ("Vanilla attention does X; CCA does Y because Z.")
  5. Connect every concept back to its purpose in a real forward pass: input_ids enter,
     logits emerge, then we sample a token, then we repeat. I should be able to trace
     a single token's path through the model at the end.

Reference materials (you can read these for context, but DO NOT modify them):

  - Architecture deep dive: `~/code/personal/zaya1-mlx/reference/notes/zaya-architecture.md`
  - Design spec: `~/code/personal/zaya1-mlx/docs/superpowers/specs/2026-05-06-zaya1-mlx-port-design.md`
  - The actual PyTorch source: `~/code/personal/zaya1-mlx/reference/.venv/lib/python3.11/site-packages/transformers/models/zaya/modular_zaya.py`
  - The MLX port code: `~/code/personal/mlx-lm/mlx_lm/models/zaya.py`

Start by asking me where I want to begin. I'll likely say "from the very beginning — what is
a token?" or similar. Then we'll go from there at whatever pace I set.

If you find yourself wanting to write code or run anything, stop and ask me first.
This is a teaching session, not an execution session.
```

---

## Notes for me (the user)

- The prompt above is intentionally long. Long context up front means fewer rounds of
  "no, I meant slower" mid-conversation.
- The "Ground rules" section is the most important part. It prevents the agent from
  defaulting back into execution mode.
- You can edit the topic list freely — add anything you're curious about, remove anything
  you already know cold.
- A natural ordering: tokens → embeddings → attention math → multi-head/GQA → norms →
  residual streams → RoPE → SwiGLU → MoE → MoD → CCA → everything ZAYA1-specific. Don't
  feel pressure to follow that exactly; whatever order makes sense to you.
- If you want to compare against another model later, the same prompt structure works.
  Replace "ZAYA1-specific" with "Llama-specific" or "Qwen-specific".

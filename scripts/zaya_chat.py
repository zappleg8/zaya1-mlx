#!/usr/bin/env python3
"""Interactive ZAYA1-8B (MLX) chat REPL.

Loads the model once at startup, then offers a streaming chat interface.
Type a message and press Enter; Ctrl-D, Ctrl-C, or 'quit' to exit.

Flags:
  --stats         Print tok/s and memory stats after each response.
  --max-tokens N  Cap response length (default: 1024).
  --model PATH    Override the model location (also ZAYA_MODEL env var).
"""
import argparse
import os
import sys


def _default_model_path():
    """Prefer local copy if present, else HF id."""
    for cand in ("ZAYA1-8B-6bit", "ZAYA1-8B-4bit", "ZAYA1-8B-bf16"):
        local = os.path.expanduser(f"~/models/{cand}")
        if os.path.isdir(local):
            return local
    return "mlx-community/ZAYA1-8B-6bit"


def main():
    ap = argparse.ArgumentParser(description="ZAYA1-8B interactive chat REPL.")
    ap.add_argument(
        "--stats",
        action="store_true",
        help="After each response, print prompt tok/s, gen tok/s, and peak memory.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per response (default: 1024).",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("ZAYA_MODEL") or _default_model_path(),
        help="Path to a local model directory or HF repo id.",
    )
    args = ap.parse_args()

    print(f"Loading {args.model} ...", file=sys.stderr, flush=True)
    from mlx_lm import load, stream_generate

    model, tokenizer = load(args.model)
    print(
        "Ready. Type a message; 'quit' or Ctrl-D to exit; 'reset' to clear history.",
        file=sys.stderr,
        flush=True,
    )
    if args.stats:
        print(
            "(stats enabled — tok/s + peak mem printed after each response)",
            file=sys.stderr,
            flush=True,
        )
    print(file=sys.stderr, flush=True)

    messages = []
    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        cmd = user_input.strip().lower()
        if cmd in ("quit", "exit"):
            return
        if cmd == "reset":
            messages = []
            print("(history cleared)\n", file=sys.stderr)
            continue
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Stream tokens; track the final GenerationResponse for stats.
        parts = []
        last_resp = None
        for resp in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=args.max_tokens
        ):
            print(resp.text, end="", flush=True)
            parts.append(resp.text)
            last_resp = resp
        print()  # newline after generation completes

        if args.stats and last_resp is not None:
            print(
                f"  [{last_resp.prompt_tokens} prompt tokens @ "
                f"{last_resp.prompt_tps:.1f} tok/s • "
                f"{last_resp.generation_tokens} gen tokens @ "
                f"{last_resp.generation_tps:.1f} tok/s • "
                f"peak {last_resp.peak_memory:.2f} GB • "
                f"finish={last_resp.finish_reason}]",
                file=sys.stderr,
                flush=True,
            )

        messages.append({"role": "assistant", "content": "".join(parts)})


if __name__ == "__main__":
    main()

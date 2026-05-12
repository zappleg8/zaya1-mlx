#!/usr/bin/env python3
"""Interactive ZAYA1-8B (MLX, 4-bit) chat REPL.

Loads the model once at startup, then offers a streaming chat interface.
Type a message and press Enter; Ctrl-D, Ctrl-C, or 'quit' to exit.

Override the model path via the ZAYA_MODEL env var.
"""
import os
import sys


def _default_model_path():
    """Prefer local copy if present, else HF id."""
    local = os.path.expanduser("~/models/ZAYA1-8B-4bit")
    if os.path.isdir(local):
        return local
    return "mlx-community/ZAYA1-8B-4bit"


def main():
    model_path = os.environ.get("ZAYA_MODEL") or _default_model_path()

    print(f"Loading {model_path} ...", file=sys.stderr, flush=True)
    from mlx_lm import load, stream_generate

    model, tokenizer = load(model_path)
    print(
        "Ready. Type a message; 'quit' or Ctrl-D to exit; 'reset' to clear history.\n",
        file=sys.stderr,
        flush=True,
    )

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

        # Stream tokens to stdout as they arrive
        parts = []
        for resp in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=1024
        ):
            print(resp.text, end="", flush=True)
            parts.append(resp.text)
        print()  # newline after generation completes

        messages.append({"role": "assistant", "content": "".join(parts)})


if __name__ == "__main__":
    main()

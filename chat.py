#!/usr/bin/env python3
from collections import deque
from config import get_openai_client, MODEL_NAME, count_tokens, compute_cost, Timer

MAX_HISTORY = 10

def chat_loop():
    client = get_openai_client()
    history = deque(maxlen=MAX_HISTORY)
    print("Chat with Assistant (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + list(history)

        prompt_tokens = count_tokens(" ".join(m["content"] for m in messages))
        completion_text = ""
        print("Assistant: ", end="", flush=True)

        with Timer() as t:
            for chunk in client.chat.completions.create(model=MODEL_NAME, messages=messages, stream=True):
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    print(delta.content, end="", flush=True)
                    completion_text += delta.content
        print()

        completion_tokens = count_tokens(completion_text)
        cost = compute_cost(prompt_tokens, completion_tokens)
        print(f"[stats] prompt={prompt_tokens} completion={completion_tokens} cost=${cost:.6f} latency={t.elapsed_ms:.0f} ms\n")
        history.append({"role": "assistant", "content": completion_text})

if __name__ == "__main__":
    chat_loop()

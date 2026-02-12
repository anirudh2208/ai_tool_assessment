#!/usr/bin/env python3
import sqlite3, os, argparse
from config import get_openai_client, MODEL_NAME, count_tokens, compute_cost, Timer

MAX_HISTORY = 10  # default, overridden by --history arg
DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")

# ── SQLite history store ────────────────────────────────────────────────────

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT)")
    return conn

def _add_message(conn, role: str, content: str):
    conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()

def _get_last_n(conn, n: int) -> list[dict]:
    rows = conn.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

# ── Chat loop ───────────────────────────────────────────────────────────────

def chat_loop(max_history: int = MAX_HISTORY):
    client = get_openai_client()
    conn = _init_db()
    existing = _get_last_n(conn, max_history)
    if existing:
        print(f"(Restored {len(existing)} messages from previous session)")
    print(f"Chat with GPT-4o — history={max_history} (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        _add_message(conn, "user", user_input)
        history = _get_last_n(conn, max_history)
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + history

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

        _add_message(conn, "assistant", completion_text)
        completion_tokens = count_tokens(completion_text)
        cost = compute_cost(prompt_tokens, completion_tokens)
        print(f"[stats] prompt={prompt_tokens} completion={completion_tokens} cost=${cost:.6f} latency={t.elapsed_ms:.0f} ms\n")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming chat with GPT-4o")
    parser.add_argument("--history", type=int, default=MAX_HISTORY, help=f"Number of messages to persist (default: {MAX_HISTORY})")
    args = parser.parse_args()
    chat_loop(max_history=args.history)
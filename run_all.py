#!/usr/bin/env python3
import subprocess, sys

TASKS = [
    ("Tests",       [sys.executable, "-m", "pytest", "tests/", "-v"]),
    ("RAG Ingest",  [sys.executable, "rag.py", "ingest"]),
    ("RAG Evaluate", [sys.executable, "rag.py", "evaluate"]),
    ("Agent",       [sys.executable, "agent.py"]),
    ("Healer",      [sys.executable, "healer.py"]),
]

def main():
    results = []
    for name, cmd in TASKS:
        print(f"\n{'═'*60}\n  Running: {name}\n{'═'*60}\n")
        r = subprocess.run(cmd, cwd="/app" if sys.platform == "linux" else ".")
        status = " PASS" if r.returncode == 0 else " FAIL"
        results.append((name, status))
        print(f"\n  → {name}: {status}")

    print(f"\n{'═'*60}\n  Summary\n{'═'*60}")
    for name, status in results:
        print(f"  {status}  {name}")

if __name__ == "__main__":
    main()

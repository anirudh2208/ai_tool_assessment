#!/usr/bin/env python3
import sys, os, re, subprocess, tempfile, shutil
from config import get_openai_client, MODEL_NAME, Timer

MAX_RETRIES = 3

# AI Assisted config
LANG_CONFIG = {
    "python": {
        "system": "Expert Python developer. Return exactly two fenced python code blocks.",
        "gen_prompt": "Return two Python code blocks: 1) solution.py 2) test_solution.py (pytest, imports from 'solution', >=3 tests).",
        "fix_prompt": "Fix the Python code. Return two code blocks: solution.py then test_solution.py.",
    },
    "rust": {
        "system": "Expert Rust developer. Return exactly one fenced rust code block.",
        "gen_prompt": "Return one Rust code block with implementation + #[cfg(test)] module with ≥3 #[test] functions. Include fn main().",
        "fix_prompt": "Fix the Rust code. Return one code block with implementation + #[cfg(test)] tests.",
    },
}

def _detect_lang(task: str) -> str:
    return "rust" if any(w in task.lower() for w in ("rust", "cargo")) else "python"

def _extract_blocks(text: str, lang: str) -> list[str]:
    for pat in [rf"```{lang}\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        blocks = re.findall(pat, text, re.DOTALL)
        if blocks: return [b.strip() for b in blocks]
    return []

def _generate(client, lang: str, task: str, error_ctx: str = None) -> list[str]:
    cfg = LANG_CONFIG[lang]
    if error_ctx:
        prompt = f"{cfg['fix_prompt']}\n\nTask: {task}\n\nErrors:\n```\n{error_ctx[-3000:]}\n```"
    else:
        prompt = f"Task: {task}\n\n{cfg['gen_prompt']}"

    print(f"  Generating {lang.title()}...", end=" ", flush=True)
    with Timer() as t:
        resp = client.chat.completions.create(model=MODEL_NAME, temperature=0.3,
            messages=[{"role": "system", "content": cfg["system"]},
                      {"role": "user", "content": prompt}])
    print(f"({t.elapsed_ms:.0f} ms)")
    return _extract_blocks(resp.choices[0].message.content, lang)

def _write_and_test_python(workdir: str, blocks: list[str]) -> tuple[bool, str]:
    if len(blocks) < 2: return False, "Expected 2 code blocks (solution + tests), got fewer."
    for name, content in [("solution.py", blocks[0]), ("test_solution.py", blocks[1])]:
        open(os.path.join(workdir, name), "w").write(content)
        print(f"  {name} ({len(content)} chars)")
    print("   Running pytest...")
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "test_solution.py", "-v", "--tb=short"],
                           capture_output=True, text=True, cwd=workdir, timeout=30)
        return r.returncode == 0, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return False, "ERROR: pytest timed out (30s)"

# AI assissted because I'm not too knowledgeable with rust
def _write_and_test_rust(workdir: str, blocks: list[str]) -> tuple[bool, str]:
    if not blocks: return False, "No code block generated."
    proj = os.path.join(workdir, "proj")
    os.makedirs(os.path.join(proj, "src"), exist_ok=True)
    open(os.path.join(proj, "Cargo.toml"), "w").write(
        '[package]\nname = "healer"\nversion = "0.1.0"\nedition = "2021"\n')
    open(os.path.join(proj, "src", "main.rs"), "w").write(blocks[0])
    print(f"  src/main.rs ({len(blocks[0])} chars)")
    for step, cmd in [(" Compiling...", ["cargo", "build"]), (" Running cargo test...", ["cargo", "test"])]:
        print(f"  {step}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=proj, timeout=60)
            if r.returncode != 0: return False, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return False, f"ERROR: {cmd[1]} timed out"
    return True, r.stdout + r.stderr

def heal(task: str) -> bool:
    client = get_openai_client()
    lang = _detect_lang(task)
    if lang == "rust" and not shutil.which("cargo"):
        print("⚠ cargo not found, falling back to Python"); lang = "python"
    workdir = tempfile.mkdtemp(prefix="healer_")
    runner = _write_and_test_rust if lang == "rust" else _write_and_test_python

    print(f"\n{'═'*60}\nSelf-Healing Code Assistant\nTask: {task}\nLang: {lang.upper()}  Dir: {workdir}\n{'═'*60}")

    error_ctx = None
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n── Attempt {attempt}/{MAX_RETRIES} {'─'*40}")
        blocks = _generate(client, lang, task, error_ctx)
        passed, output = runner(workdir, blocks)
        for line in output.strip().split("\n"): print(f"     {line}")

        if passed:
            print(f"\n{'═'*60}\n  ALL TESTS PASSED (attempt {attempt})!\n{'═'*60}")
            print(f"\n── Code {'─'*51}\n{blocks[0]}")
            return True
        print(f"\n  Failed attempt {attempt}")
        error_ctx = output

    print(f"\n{'═'*60}\n  FAILED after {MAX_RETRIES} attempts\n{'═'*60}")
    return False

if __name__ == "__main__":
    heal(" ".join(sys.argv[1:]) or "write a function called quicksort that sorts a list using the quicksort algorithm")
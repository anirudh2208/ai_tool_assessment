import os, sys, time, pytest
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

#  Config tests 
class TestConfig:
    def test_count_tokens(self):
        from config import count_tokens
        assert count_tokens("hello world") > 0
        assert count_tokens("") == 0

    def test_compute_cost(self):
        from config import compute_cost
        # 100 prompt tokens @ $0.0025/1K + 50 completion tokens @ $0.01/1K
        assert compute_cost(100, 50) == pytest.approx(0.00025 + 0.0005, rel=1e-3)

    def test_timer(self):
        from config import Timer
        with Timer() as t: time.sleep(0.01)
        assert t.elapsed_ms >= 10

#  Chat tests 
class TestChat:
    def test_deque_bounded(self):
        h = deque(maxlen=10)
        for i in range(20): h.append(f"msg{i}")
        assert len(h) == 10 and h[0] == "msg10"

#  RAG tests
class TestRAG:
    def test_min_20_questions(self):
        from rag import EVAL_QUESTIONS
        assert len(EVAL_QUESTIONS) >= 20

    def test_question_fields(self):
        from rag import EVAL_QUESTIONS
        for q in EVAL_QUESTIONS:
            assert "q" in q and "keywords" in q and len(q["keywords"]) > 0

#  Agent tool tests 
class TestAgentTools:
    def test_flights(self):
        from agent import search_flights
        r = search_flights("Wellington", "Auckland", "2025-03-15")
        assert len(r) > 0 and all("price_nzd" in f for f in r)

    def test_flights_filter(self):
        from agent import search_flights
        assert all(f["price_nzd"] <= 60 for f in search_flights("A", "B", "2025-01-01", 60))

    def test_weather(self):
        from agent import get_weather
        w = get_weather("Auckland", "2025-03-15")
        assert all(k in w for k in ("condition", "high_c", "low_c"))

    def test_attractions(self):
        from agent import search_attractions
        assert len(search_attractions("Auckland")) > 0

    def test_attractions_filter(self):
        from agent import search_attractions
        assert all(a["category"] == "free" for a in search_attractions("Auckland", "free"))

    def test_accommodation(self):
        from agent import search_accommodation
        assert len(search_accommodation("Auckland", "2025-03-15", 2)) > 0

    def test_accommodation_filter(self):
        from agent import search_accommodation
        assert all(h["price_per_night_nzd"]*2 <= 80 for h in search_accommodation("Auckland", "2025-03-15", 2, 80))

    def test_dispatch_matches_spec(self):
        from agent import TOOL_DISPATCH, TOOLS_SPEC
        assert {t["function"]["name"] for t in TOOLS_SPEC} == set(TOOL_DISPATCH.keys())

#  Healer tests 
class TestHealer:
    def test_extract_blocks_python(self):
        from healer import _extract_blocks
        text = "here:\n```python\ndef hello(): pass\n```\nand:\n```python\ndef test(): pass\n```"
        blocks = _extract_blocks(text, "python")
        assert len(blocks) == 2
        assert "def hello" in blocks[0]

    # AI Generated (only this test)
    def test_extract_blocks_rust(self):
        from healer import _extract_blocks
        text = "```rust\nfn main() {}\n```"
        blocks = _extract_blocks(text, "rust")
        assert len(blocks) == 1
        assert "fn main" in blocks[0]

    def test_detect_lang_python(self):
        from healer import _detect_lang
        assert _detect_lang("write quicksort in Python") == "python"
        assert _detect_lang("write a palindrome checker") == "python"

    def test_detect_lang_rust(self):
        from healer import _detect_lang
        assert _detect_lang("write quicksort in Rust") == "rust"
        assert _detect_lang("implement binary search with cargo test") == "rust"

    def test_max_retries(self):
        from healer import MAX_RETRIES
        assert MAX_RETRIES == 3

#  API tests
class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from api import app
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
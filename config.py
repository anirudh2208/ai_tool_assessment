import os, time, tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Gpt4o")
CHROMA_HOST = os.getenv("CHROMA_HOST", "")  # empty = embedded, set to "chromadb" in Docker (For future)

# GPT-4o pricing per 1K tokens (from $2.50/1M input, $10.00/1M output) from OpenAI website
COST_PER_1K_PROMPT = 0.0025
COST_PER_1K_COMPLETION = 0.01

_enc = tiktoken.encoding_for_model("gpt-4o") # Since give model is gpt-4o in the task document (will change if model changes)

def get_openai_client() -> OpenAI:
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1000) * COST_PER_1K_PROMPT + (completion_tokens / 1000) * COST_PER_1K_COMPLETION

class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
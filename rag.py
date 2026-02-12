#!/usr/bin/env python3
import sys, os, hashlib, requests
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config import OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME, CHROMA_HOST, Timer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION = "rag_docs"

# AI Generated sources
SOURCES = [
    # Fellowship of the Ring (~2.5 MB)
    {"url": "https://www.mrsmuellersworld.com/uploads/1/3/0/5/13054185/lord-of-the-rings-01-the-fellowship-of-the-ring_full_text.pdf",
     "filename": "fellowship_of_the_ring.pdf"},
    # War and Peace by Tolstoy (~4 MB, public domain)
    {"url": "https://www.planetebook.com/free-ebooks/war-and-peace.pdf",
     "filename": "war_and_peace.pdf"},
    # War and Peace - Internet Archive edition (~20 MB, public domain)
    {"url": "https://archive.org/download/war-peace/war-peace.pdf",
     "filename": "war_and_peace_illustrated.pdf"},
    # War and Peace - public-library.uk (~17 MB, public domain)
    {"url": "http://public-library.uk/dailyebook/Count%20Leo%20Tolstoy%20War%20and%20peace%20(1900).pdf",
     "filename": "war_and_peace_1900.pdf"},
    # Complete Works of Shakespeare from Gutenberg (~5.5 MB)
    {"url": "https://www.gutenberg.org/files/100/100-0.txt",
     "filename": "complete_shakespeare.txt"},
    # Drop any extra PDFs into data/ to add more corpus
]

# Chroma's built-in embedding (onnxruntime + all-MiniLM-L6-v2, no PyTorch needed) fixed it from an error
_default_ef = DefaultEmbeddingFunction()

class _ChromaEmbeddingAdapter:
    """Wraps Chroma's DefaultEmbeddingFunction to match LangChain's Embeddings interface."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _default_ef(texts)
    def embed_query(self, text: str) -> list[float]:
        return _default_ef([text])[0]

def _embeddings():
    return _ChromaEmbeddingAdapter()

def _chroma_client():
    if CHROMA_HOST:
        return chromadb.HttpClient(host=CHROMA_HOST, port=8000)
    return None

def _vectorstore():
    client = _chroma_client()
    if client:
        return Chroma(client=client, embedding_function=_embeddings(), collection_name=COLLECTION)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings(), collection_name=COLLECTION)

# Ingest
def ingest():
    os.makedirs(DATA_DIR, exist_ok=True)
    for src in SOURCES:
        dest = os.path.join(DATA_DIR, src["filename"])
        if os.path.exists(dest):
            print(f"  {src['filename']} exists"); continue
        print(f"  Downloading {src['filename']}...")
        try:
            r = requests.get(src["url"], timeout=120); r.raise_for_status()
            open(dest, "wb").write(r.content)
            print(f"    Saved ({len(r.content)/1e6:.1f} MB)")
        except Exception as e:
            print(f"    Failed: {e}. Drop PDF manually into {DATA_DIR}/")

    docs = []
    for f in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, f)
        if f.lower().endswith(".pdf"):
            print(f"  Loading {f}...")
            docs.extend(PyPDFLoader(fpath).load())
        elif f.lower().endswith(".txt"):
            print(f"  Loading {f}...")
            from langchain_community.document_loaders import TextLoader
            docs.extend(TextLoader(fpath, encoding="utf-8").load())
    print(f"  Pages/docs loaded: {len(docs)}")

    # Larger chunks + more overlap = better context preservation for retrieval
    chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300).split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = hashlib.md5(f"{c.metadata.get('source','')}-{i}".encode()).hexdigest()[:12]
        # Storing a clean source name for citations
        src_path = c.metadata.get("source", "unknown")
        c.metadata["source_name"] = os.path.basename(src_path).replace("_", " ").replace(".pdf", "")

    total = len(chunks)
    batch_size = 100
    print(f"  Embedding {total} chunks (local model: all-MiniLM-L6-v2)...")

    from tqdm import tqdm
    emb = _embeddings()
    client = _chroma_client()

    # First batch creates the store, subsequent batches add to it
    for i in tqdm(range(0, total, batch_size), desc="  Embedding", unit="batch"):
        batch = chunks[i:i + batch_size]
        if i == 0:
            if client:
                db = Chroma.from_documents(batch, emb, client=client, collection_name=COLLECTION)
            else:
                db = Chroma.from_documents(batch, emb, persist_directory=PERSIST_DIR, collection_name=COLLECTION)
        else:
            db.add_documents(batch)

    print(f"   Done — {total} chunks embedded")

# ── Query ───────────────────────────────────────────────────────────────────
def query(question: str, top_k: int = 5, verbose: bool = True) -> dict:
    db = _vectorstore()
    with Timer() as rt:
        results = db.similarity_search_with_score(question, k=top_k)
    if verbose: print(f"[retrieval] {rt.elapsed_ms:.0f} ms")

    # Build context with detailed citation markers
    context_parts = []
    for i, (d, _) in enumerate(results, 1):
        src_name = d.metadata.get("source_name", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "?")
        context_parts.append(f"[{i}] — {src_name}, p.{page}\n{d.page_content}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a QA assistant. Answer using ONLY the provided context.\n"
        "For EVERY claim, include an inline citation with the source name and page number, "
        "formatted like: [1: Fellowship of the Ring, p.42]. "
        "Use the citation labels from the context. If the answer is not in the context, say so."
    )

    llm = ChatOpenAI(openai_api_base=OPENAI_BASE_URL, openai_api_key=OPENAI_API_KEY, model=MODEL_NAME, temperature=0)
    with Timer() as lt:
        resp = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ])
    if verbose: print(f"[llm] {lt.elapsed_ms:.0f} ms\n\nAnswer: {resp.content}\n")
    return {"answer": resp.content, "retrieval_ms": rt.elapsed_ms, "llm_ms": lt.elapsed_ms,
            "sources": [{"source": d.metadata.get("source_name",""), "page": d.metadata.get("page",""), "score": float(s)} for d,s in results]}

# ── Evaluation (≥20 graded questions) ──────────────────────────────────────
# Keywords are lowercased substrings that should appear in the top-K retrieved chunks.
# AI Generated Questions
EVAL_QUESTIONS = [
    {"q": "What is the name of Frodo's sword?", "keywords": ["sting"]},
    {"q": "Who leads the Fellowship of the Ring on their quest?", "keywords": ["aragorn", "gandalf", "strider"]},
    {"q": "What is written on the One Ring in the Black Speech?", "keywords": ["one ring to rule", "ash nazg"]},
    {"q": "What is the name of the Elven city where the Council is held?", "keywords": ["rivendell", "imladris"]},
    {"q": "Who is Sam Gamgee and what is his role?", "keywords": ["sam", "samwise", "gardener", "servant"]},
    {"q": "What is the Balrog and where does the Fellowship encounter it?", "keywords": ["balrog", "flame", "shadow", "moria"]},
    {"q": "Who is Sauron the Dark Lord?", "keywords": ["sauron", "dark lord", "enemy"]},
    {"q": "What gift does Bilbo give Frodo at Rivendell?", "keywords": ["bilbo", "mithril", "sting", "coat", "mail"]},
    {"q": "What is the name of Gandalf's horse?", "keywords": ["shadowfax", "horse", "steed"]},
    {"q": "Who is Boromir son of Denethor?", "keywords": ["boromir", "gondor", "denethor"]},
    {"q": "Describe the Shire where the hobbits live", "keywords": ["shire", "hobbit", "hobbits", "hobbiton"]},
    {"q": "How was the One Ring created and by whom?", "keywords": ["sauron", "forge", "ring", "power", "mount doom", "orodruin"]},
    {"q": "Who is the elf Legolas in the Fellowship?", "keywords": ["legolas", "elf", "mirkwood", "thranduil"]},
    {"q": "Who is the dwarf Gimli in the Fellowship?", "keywords": ["gimli", "dwarf", "gloin"]},
    {"q": "What is the story of Gollum and his real name?", "keywords": ["gollum", "smeagol", "sméagol", "precious"]},
    {"q": "Who is Arwen Evenstar and her relationship with Aragorn?", "keywords": ["arwen", "evenstar", "elrond"]},
    {"q": "What does Galadriel show Frodo in her mirror?", "keywords": ["mirror", "galadriel", "water", "vision"]},
    {"q": "What great river does the Fellowship travel along?", "keywords": ["anduin", "great river", "river"]},
    {"q": "What happens when Gandalf fights the Balrog on the Bridge?", "keywords": ["bridge", "khazad", "durin", "balrog", "fell", "fly"]},
    {"q": "What are the Nazgul or Ringwraiths?", "keywords": ["nazgul", "ringwraith", "black rider", "nine"]},
]

def evaluate():
    db = _vectorstore()
    correct, times = 0, []
    print(f"Evaluating {len(EVAL_QUESTIONS)} questions...\n")
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        with Timer() as t:
            results = db.similarity_search(item["q"], k=10)  # wider net for eval
        times.append(t.elapsed_ms)
        combined = " ".join(d.page_content.lower() for d in results)
        hit = any(kw in combined for kw in item["keywords"])
        if hit: correct += 1
        print(f"  {'Yes' if hit else 'No'} Q{i}: {item['q'][:55]}  ({t.elapsed_ms:.0f} ms)")

    print(f"\n{'='*50}")
    print(f"Top-10 Retrieval Accuracy: {correct}/{len(EVAL_QUESTIONS)} ({correct/len(EVAL_QUESTIONS)*100:.1f}%)")
    print(f"Median retrieval time:     {sorted(times)[len(times)//2]:.0f} ms")
    print(f"{'='*50}")

# CLI, Multi-modes
if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd == "ingest":     ingest()
    elif cmd == "query":    query(" ".join(sys.argv[2:]) or input("Question: "))
    elif cmd == "evaluate": evaluate()
    else: print("Usage: python rag.py [ingest|query|evaluate]")
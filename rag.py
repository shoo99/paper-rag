#!/usr/bin/env python3
"""
paper-rag — a tiny, fully-local RAG over your own PDFs.

Stack (all local, nothing leaves the machine):
  BGE-M3 embeddings (via Ollama) + Qdrant (embedded, no server) + a local LLM (via Ollama) + pypdf.

Quick start:
  pip install pypdf qdrant-client
  ollama pull bge-m3
  ollama pull qwen3:8b          # or any chat model; set RAG_LLM to override
  python rag.py ingest ./papers # a folder of PDFs (or individual files)
  python rag.py ask "your question"

Config via env (all optional):
  OLLAMA_URL  (default http://127.0.0.1:11434)
  RAG_EMBED   (default bge-m3)
  RAG_LLM     (default qwen3:8b)   — any Ollama chat model
  RAG_DB      (default ./rag_qdrant)
"""
import os, sys, json, glob, re, time, urllib.request, hashlib

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

OLLAMA      = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED", "bge-m3")          # 1024-dim, multilingual
LLM_MODEL   = os.environ.get("RAG_LLM",   "qwen3:8b")        # any Ollama chat model
QDRANT_PATH = os.environ.get("RAG_DB",    "./rag_qdrant")
COLLECTION  = "papers"
DIM         = 1024
CHUNK, OVERLAP, TOPK = 1400, 200, 5


def _post(path, payload, timeout=600):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    return _post("/api/embed", {"model": EMBED_MODEL, "input": texts})["embeddings"]


def llm(system, user):
    # /api/chat + think:false works across models and skips reasoning-model chain-of-thought.
    out = _post("/api/chat", {
        "model": LLM_MODEL, "think": False, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"num_predict": 600, "temperature": 0.2},
    })
    return out.get("message", {}).get("content", "").strip()


def client():
    c = QdrantClient(path=QDRANT_PATH)
    if COLLECTION not in [x.name for x in c.get_collections().collections]:
        c.create_collection(COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    return c


def chunk(text):
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + CHUNK]); i += CHUNK - OVERLAP
    return [c for c in out if len(c) > 60]


def ingest(paths):
    files = []
    for p in paths:
        files += glob.glob(os.path.join(p, "*.pdf")) if os.path.isdir(p) else [p]
    c, pts, n = client(), [], 0
    for f in files:
        name = os.path.basename(f)
        try:
            pages = [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(PdfReader(f).pages)]
        except Exception as e:
            print(f"  ! skip {name}: {e}"); continue
        chunks = [(pg, ch) for pg, t in pages for ch in chunk(t)]
        if not chunks:
            print(f"  ! {name}: no extractable text (scanned PDF?)"); continue
        for b in range(0, len(chunks), 16):
            batch = chunks[b:b + 16]
            for (pg, ch), v in zip(batch, embed([ch for _, ch in batch])):
                pid = int(hashlib.md5(f"{name}:{pg}:{ch[:40]}:{b}".encode()).hexdigest()[:15], 16)
                pts.append(PointStruct(id=pid, vector=v, payload={"text": ch, "source": name, "page": pg}))
            n += len(batch)
        print(f"  + {name}: {len(chunks)} chunks")
    for b in range(0, len(pts), 256):
        c.upsert(COLLECTION, pts[b:b + 256])
    print(f"Indexed {n} chunks from {len(files)} file(s) -> {QDRANT_PATH}")


def ask(question):
    c = client()
    hits = c.query_points(COLLECTION, query=embed(question)[0], limit=TOPK).points
    if not hits:
        print("No matches - ingest some PDFs first."); return
    ctx = "\n\n".join(f"[{i+1}] ({h.payload['source']} p.{h.payload['page']}) {h.payload['text']}"
                      for i, h in enumerate(hits))
    system = ("You are a research assistant. Answer using ONLY the provided context excerpts. "
              "Cite sources inline as [n]. If the context lacks the answer, say so plainly.")
    t = time.time()
    print("\n" + llm(system, f"CONTEXT:\n{ctx}\n\nQUESTION: {question}") + "\n\nSources:")
    for i, h in enumerate(hits):
        print(f"  [{i+1}] {h.payload['source']} p.{h.payload['page']}  (score {h.score:.3f})")
    print(f"\n({time.time()-t:.1f}s, fully local - nothing left this machine)")


def stats():
    info = client().get_collection(COLLECTION)
    print(f"Collection '{COLLECTION}': {info.points_count} chunks, dim {DIM}, at {QDRANT_PATH}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ingest":   ingest(sys.argv[2:])
    elif cmd == "ask":    ask(" ".join(sys.argv[2:]))
    elif cmd == "stats":  stats()
    else:                 print(__doc__)

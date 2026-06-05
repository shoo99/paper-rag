#!/usr/bin/env python3
"""
paper-rag — a tiny, fully-local RAG over your own PDFs (hybrid search + reranking).

Stack (all local, nothing leaves the machine):
  BGE-M3 dense embeddings (via Ollama) + BM25 sparse (fastembed) + Qdrant (embedded)
  + a cross-encoder reranker (fastembed) + a local LLM (via Ollama) + pypdf.

Retrieval: dense and sparse hits are fused (Reciprocal Rank Fusion) into a candidate
pool, then a cross-encoder reranks them and the best passages go to the LLM for a cited
answer. If fastembed isn't installed it falls back to dense-only (no sparse, no rerank).

Quick start:
  pip install pypdf qdrant-client fastembed
  ollama pull bge-m3
  ollama pull qwen3:8b          # or any chat model; set RAG_LLM to override
  python rag.py ingest ./papers
  python rag.py ask "your question"

Config via env (all optional):
  OLLAMA_URL  (default http://127.0.0.1:11434)  — point at any host running Ollama
  RAG_EMBED   (default bge-m3)
  RAG_LLM     (default qwen3:8b)
  RAG_DB      (default ./rag_qdrant)
  RAG_RERANK  (default Xenova/ms-marco-MiniLM-L-6-v2) — any fastembed cross-encoder
  RAG_SPARSE  (default Qdrant/bm25)
  RAG_NUM_CTX (default 8192)  — caps the LLM context so big-native-context models stay on-GPU
  RAG_EMBED_TIMEOUT (default 120)

Ingest is resumable: progress is saved per batch, so re-running picks up where it stopped.
"""
import os, sys, json, glob, re, time, urllib.request, hashlib

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector, Prefetch, FusionQuery, Fusion)

OLLAMA       = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL  = os.environ.get("RAG_EMBED", "bge-m3")          # 1024-dim, multilingual
LLM_MODEL    = os.environ.get("RAG_LLM",   "qwen3:8b")        # any Ollama chat model
QDRANT_PATH  = os.environ.get("RAG_DB",    "./rag_qdrant")
RERANK_MODEL = os.environ.get("RAG_RERANK","Xenova/ms-marco-MiniLM-L-6-v2")
SPARSE_MODEL = os.environ.get("RAG_SPARSE","Qdrant/bm25")
NUM_CTX      = int(os.environ.get("RAG_NUM_CTX", "8192"))     # cap KV cache -> stay fully on GPU
EMBED_TIMEOUT= int(os.environ.get("RAG_EMBED_TIMEOUT", "120"))
COLLECTION   = "papers"
DIM          = 1024
CHUNK, OVERLAP, BATCH = 1400, 200, 16
PREFETCH, TOPK = 20, 5          # hybrid candidate pool -> rerank -> passages sent to the LLM
DENSE, SPARSE  = "dense", "sparse"

# Optional hybrid (sparse retrieval + cross-encoder rerank); degrade to dense-only if absent.
try:
    from fastembed import SparseTextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    HYBRID = True
except Exception:
    HYBRID = False

_sparse = _reranker = None
def sparse_model():
    global _sparse
    if _sparse is None: _sparse = SparseTextEmbedding(SPARSE_MODEL)
    return _sparse
def reranker():
    global _reranker
    if _reranker is None: _reranker = TextCrossEncoder(RERANK_MODEL)
    return _reranker
def _sv(o):                                       # fastembed sparse output -> Qdrant SparseVector
    return SparseVector(indices=[int(i) for i in o.indices], values=[float(v) for v in o.values])


def _post(path, payload, timeout=600):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(texts, tries=3):
    if isinstance(texts, str):
        texts = [texts]
    for k in range(tries):                       # embedders can stall under load; fail fast + retry, don't hang
        try:
            return _post("/api/embed", {"model": EMBED_MODEL, "input": texts}, timeout=EMBED_TIMEOUT)["embeddings"]
        except Exception as e:
            if k == tries - 1:
                raise SystemExit(
                    f"\nEmbedding failed after {tries} tries: {e}\n"
                    f"'{EMBED_MODEL}' is likely stuck (a known Ollama hiccup under sustained load).\n"
                    f"Fix: `ollama stop {EMBED_MODEL}` and re-run; on WSL, `wsl --shutdown` then re-run.\n"
                    f"Progress is saved per batch — re-running the same ingest resumes where it stopped.")
            time.sleep(2 * (k + 1))


def llm(system, user):
    # /api/chat + think:false works across models and skips reasoning-model chain-of-thought.
    out = _post("/api/chat", {
        "model": LLM_MODEL, "think": False, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"num_predict": 600, "temperature": 0.2, "num_ctx": NUM_CTX},
    })
    return out.get("message", {}).get("content", "").strip()


_qc = None
def client():
    global _qc
    if _qc is None:                                   # cache one embedded client (reused across MCP calls)
        _qc = QdrantClient(path=QDRANT_PATH)
        if COLLECTION not in [x.name for x in _qc.get_collections().collections]:
            _qc.create_collection(COLLECTION,
                vectors_config={DENSE: VectorParams(size=DIM, distance=Distance.COSINE)},
                sparse_vectors_config={SPARSE: SparseVectorParams()})   # present even in dense-only mode
    return _qc


def chunk(text):
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + CHUNK]); i += CHUNK - OVERLAP
    return [c for c in out if len(c) > 60]


def _pid(name, pg, ch, b):                        # deterministic id -> upserts are idempotent, so ingest can resume
    return int(hashlib.md5(f"{name}:{pg}:{ch[:40]}:{b}".encode()).hexdigest()[:15], 16)


def ingest(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.pdf"))) if os.path.isdir(p) else [p]
    c, n, fresh = client(), 0, 0
    sm = sparse_model() if HYBRID else None
    for f in files:
        name = os.path.basename(f)
        try:
            pages = [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(PdfReader(f).pages)]
        except Exception as e:
            print(f"  ! skip {name}: {e}"); continue
        chunks = [(pg, ch) for pg, t in pages for ch in chunk(t)]
        if not chunks:
            print(f"  ! {name}: no extractable text (scanned PDF?)"); continue
        for b in range(0, len(chunks), BATCH):
            batch = chunks[b:b + BATCH]
            ids = [_pid(name, pg, ch, b) for pg, ch in batch]
            if len(c.retrieve(COLLECTION, ids=ids)) == len(ids):     # resume: batch already stored -> skip
                n += len(batch); continue
            texts = [ch for _, ch in batch]
            dvecs = embed(texts)
            svecs = list(sm.embed(texts)) if HYBRID else [None] * len(texts)
            pts = []
            for i, (pg, ch), dv, sv in zip(ids, batch, dvecs, svecs):
                vec = {DENSE: dv}
                if HYBRID: vec[SPARSE] = _sv(sv)
                pts.append(PointStruct(id=i, vector=vec, payload={"text": ch, "source": name, "page": pg}))
            c.upsert(COLLECTION, pts)                                 # persist each batch -> a stall never loses earlier work
            n += len(batch); fresh += len(batch)
        print(f"  + {name}: {len(chunks)} chunks")
    print(f"Indexed {n} chunks from {len(files)} file(s) ({fresh} newly embedded) -> {QDRANT_PATH}"
          + ("  [hybrid: dense+sparse+rerank]" if HYBRID else "  [dense-only: pip install fastembed for hybrid+rerank]"))


def retrieve(question, k=TOPK):
    """Top-k passages for a question (hybrid dense+sparse + rerank, or dense-only). Returns a list of dicts."""
    c = client()
    if HYBRID:
        qd = embed(question)[0]
        qs = _sv(next(sparse_model().query_embed(question)))
        cand = c.query_points(COLLECTION,                            # dense + sparse, fused by RRF
            prefetch=[Prefetch(query=qd, using=DENSE,  limit=PREFETCH),
                      Prefetch(query=qs, using=SPARSE, limit=PREFETCH)],
            query=FusionQuery(fusion=Fusion.RRF), limit=PREFETCH).points
        if not cand:
            return []
        scores = list(reranker().rerank(question, [h.payload["text"] for h in cand]))   # cross-encoder rerank
        ranked = sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)[:k]
    else:
        pts = c.query_points(COLLECTION, query=embed(question)[0], using=DENSE, limit=k).points
        ranked = [(h, h.score) for h in pts]
    return [{"text": h.payload["text"], "source": h.payload["source"],
             "page": h.payload["page"], "score": round(float(s), 4)} for h, s in ranked]


def answer(question):
    """Retrieve, then have the local LLM answer with [n] citations. Returns {'answer': str, 'sources': [...]}."""
    hits = retrieve(question)
    if not hits:
        return {"answer": "No matches - ingest some PDFs first.", "sources": []}
    ctx = "\n\n".join(f"[{i+1}] ({h['source']} p.{h['page']}) {h['text']}" for i, h in enumerate(hits))
    system = ("You are a research assistant. Answer using ONLY the provided context excerpts. "
              "Cite sources inline as [n]. If the context lacks the answer, say so plainly.")
    ans = llm(system, f"CONTEXT:\n{ctx}\n\nQUESTION: {question}")
    sources = [{"n": i + 1, "source": h["source"], "page": h["page"], "score": h["score"]}
               for i, h in enumerate(hits)]
    return {"answer": ans, "sources": sources}


def ask(question):
    t = time.time()
    r = answer(question)
    print("\n" + r["answer"] + "\n\nSources:")
    for s in r["sources"]:
        print(f"  [{s['n']}] {s['source']} p.{s['page']}  ({'rerank' if HYBRID else 'score'} {s['score']:.3f})")
    print(f"\n({time.time()-t:.1f}s, {'hybrid+rerank' if HYBRID else 'dense'}, fully local - nothing left this machine)")


def stats():
    info = client().get_collection(COLLECTION)
    print(f"Collection '{COLLECTION}': {info.points_count} chunks, dim {DIM}, at {QDRANT_PATH} "
          f"({'hybrid' if HYBRID else 'dense-only'})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ingest":   ingest(sys.argv[2:])
    elif cmd == "ask":    ask(" ".join(sys.argv[2:]))
    elif cmd == "stats":  stats()
    else:                 print(__doc__)

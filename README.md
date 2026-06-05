# paper-rag — a private, fully-local RAG over your own papers

Ask questions about your own PDF library in natural language, get **cited** answers, **fully offline** — nothing leaves your machine. **Hybrid retrieval (dense + sparse) with cross-encoder reranking**, ~230 lines of Python.

Built for researchers who can't (or won't) send their corpus to a cloud API. Runs on a single modest GPU, or CPU-only.

> The capstone project of [**Local LLMs for Researchers**](https://github.com/shoo99/local-llm-for-researchers).

## Why

A vector search *finds* passages; it can't *reason* over them. This is exhaustive, **cited** question-answering over papers you own — with the whole stack (embeddings, sparse index, reranker, vector DB, LLM) running locally:

```
PDFs ─▶ chunk ─┬─▶ BGE-M3 dense  (Ollama)  ─┐
               └─▶ BM25 sparse (fastembed) ─┴─▶ Qdrant (embedded, on disk)
                                                     │
question ─▶ dense + sparse ─▶ RRF fusion ─▶ cross-encoder rerank ─▶ top passages
                                                                         │
                                                                         ▼
                                                   local LLM (Ollama) ─▶ cited answer
```

Dense catches *meaning*, sparse (BM25) catches *exact terms* (gene names, identifiers); fusing them and then reranking with a cross-encoder gives the LLM noticeably better context than dense-alone.

## Quick start (~5 minutes)

```bash
pip install pypdf qdrant-client fastembed   # fastembed is optional (see below)
ollama pull bge-m3          # dense embeddings
ollama pull qwen3:8b        # or any chat model; set RAG_LLM to override

python rag.py ingest ./papers          # a folder of PDFs (or individual files)
python rag.py ask "What method did paper X use for batch correction?"
python rag.py stats
```

That's it. No server to run, no API key, no data leaving the box. `fastembed` (BM25 sparse + the reranker) is optional — without it, retrieval runs **dense-only** and everything else works the same. Upgrading from an older dense-only index? The vector schema changed — delete `rag_qdrant/` and re-ingest once.

## Example

```
$ python rag.py ask "Which gene passed all three colocalization tests, and why was DRD2 demoted?"

SLC12A5 (KCC2) was the only gene that passed all three tests (SMR, HEIDI, and COLOC) [1].
DRD2 was demoted because it did not colocalize (COLOC PP4 ≈ 0): the GWAS and eQTL signals
point to distinct variants, so the association is likely LD-driven, not cis-causal [1][3].

Sources:
  [1] paper.pdf p.1   (rerank 4.054)
  [2] paper.pdf p.10  (rerank 3.938)
  ...
(14s, hybrid+rerank, fully local - nothing left this machine)
```

## How it works

- **Chunking** — PDFs → text (`pypdf`) → ~1400-char overlapping chunks, tagged with source + page.
- **Embeddings** — BGE-M3 dense via Ollama (`/api/embed`), 1024-dim, multilingual, **plus BM25 sparse** (fastembed) for exact-term matching.
- **Storage** — **Qdrant in embedded mode** (`QdrantClient(path=...)`) with **named dense + sparse vectors** — a real vector DB, no Docker, no server, just a folder on disk.
- **Retrieval** — **hybrid**: dense and sparse top-20 are fused with **Reciprocal Rank Fusion**, then a **cross-encoder reranker** (fastembed, CPU) reorders them, and the best 5 passages go to the LLM.
- **Answer** — a local LLM answers *only* from the retrieved context with `[n]` citations.
- **Fallback** — no `fastembed`? It runs **dense-only** (cosine top-k, no sparse, no rerank); same CLI, same output.

## Config (all optional, via env)

| Var | Default | |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `RAG_EMBED` | `bge-m3` | embedding model |
| `RAG_LLM` | `qwen3:8b` | any Ollama chat model |
| `RAG_DB` | `./rag_qdrant` | vector DB folder |
| `RAG_RERANK` | `Xenova/ms-marco-MiniLM-L-6-v2` | any fastembed cross-encoder (e.g. `jinaai/jina-reranker-v2-base-multilingual`, `BAAI/bge-reranker-base`) |
| `RAG_SPARSE` | `Qdrant/bm25` | fastembed sparse model |
| `RAG_NUM_CTX` | `8192` | LLM context cap — keeps big-native-context models fully on-GPU instead of spilling KV cache to CPU |
| `RAG_EMBED_TIMEOUT` | `120` | seconds before an embed request is retried |

## Notes from building it (the things that bit me)

- **Reasoning models hide the answer.** If your LLM emits a `<think>` block, its answer can come back empty. This uses `"think": false` so you get the final answer, not the chain-of-thought.
- **Qdrant's API moved.** Recent `qdrant-client` uses `query_points()`, not the old `search()`.
- **Embedded Qdrant is underrated** — you get the real engine without standing up a server; perfect for a single-machine private tool.
- **Embedders can stall under load.** On some setups (notably older GPUs and WSL) a long ingest can make the embedding model hang. So `ingest` saves progress **per batch** with deterministic IDs: if it stalls, re-run the same command and it resumes — and a stuck embed call fails fast (see [Troubleshooting](#troubleshooting)) instead of hanging forever.
- **A huge native context silently halves your speed.** Some models ship a giant native context (e.g. 256K); Ollama then allocates a KV cache to match, which can overflow VRAM and spill to CPU — on a 24 GB card a 27B model dropped from ~36 to ~17 tok/s. `RAG_NUM_CTX` (default 8192) caps it so the model stays fully on-GPU. RAG prompts are small, so a modest context loses nothing.

## Troubleshooting

**`ingest` hangs, or an embed call times out.** The embedding model (in Ollama) has stalled — a known hiccup under sustained load, especially on older GPUs or under WSL. Fix:

```bash
ollama stop bge-m3                # unload the stuck model
python rag.py ingest ./papers     # re-run the SAME ingest — it resumes where it stopped
```

On **WSL**, if Ollama or even `nvidia-smi` won't respond at all (the GPU is wedged), reset the VM from Windows PowerShell, then re-run the ingest:

```powershell
wsl --shutdown
```

Nothing is lost — already-embedded batches are on disk, so the re-run only embeds what's left.

**If it keeps wedging, run the embedder on CPU** (the real fix on some setups — notably older Pascal GPUs under WSL, where the *embedding* model can repeatedly hang the GPU). The embedder is tiny (~1 GB), so pin it to CPU and keep your LLM on the GPU:

```bash
printf 'FROM bge-m3\nPARAMETER num_gpu 0\n' > bge-m3-cpu.Modelfile
ollama create bge-m3-cpu -f bge-m3-cpu.Modelfile
RAG_EMBED=bge-m3-cpu python rag.py ingest ./papers     # embeddings on CPU, answers still on the GPU
```

CPU embedding is plenty fast for a personal library (~100 chunks in well under a minute) and sidesteps the GPU hang entirely.

## Honest limitations

- Naive fixed-size chunking (no semantic/late chunking yet). Good enough for "find and answer," not a production search system.
- **Reranking runs on CPU** (fastembed/ONNX) — fast for a personal library (reranking ~20 candidates is sub-second to a couple of seconds); for a very large corpus a GPU rerank server (TEI/Infinity) would scale better.
- Answer quality is your local model's quality — verify domain-specific claims (a small local model can be fluent and wrong).
- Text-based PDFs only (scanned/image PDFs need OCR first).

## License

MIT.

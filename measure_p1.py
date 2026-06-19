#!/usr/bin/env python3
"""Measure paper-rag v2 P1 (dedup + budget) before/after, on the ingested test corpus.
Run with: RAG_DB=./rag_qdrant_test RAG_EMBED=bge-m3-cpu RAG_LLM=qwen3:8b python measure_p1.py"""
import time, rag

QUESTIONS = [
    "Can flux balance analysis predict antibiotic synergy in ESKAPE pathogens?",
    "What machine-learning alternative is proposed for predicting drug combination synergy?",
    "What are the limitations of genome-scale metabolic models for synergy prediction?",
    "How does chronic stress affect the brain according to the scoping review?",
    "Which brain regions are most implicated in chronic stress?",
]

def run(label, rank_k, dedup, budget):
    rag.RANK_K, rag.DEDUP, rag.CTX_BUDGET = rank_k, dedup, budget
    print(f"\n=== {label}  (RANK_K={rank_k}, DEDUP={dedup}, BUDGET={budget}) ===")
    tot_chars, tot_pass, tot_raw, tot_dropped = 0, 0, 0, 0
    for q in QUESTIONS:
        raw = rag.retrieve(q, k=rank_k)
        after = rag.curate(q)
        n_dedup = len(rag._dedup(raw)) if dedup else len(raw)
        dropped = len(raw) - n_dedup
        chars = sum(len(h["text"]) for h in after)
        srcs = len(set(h["source"] for h in after))
        tot_chars += chars; tot_pass += len(after); tot_raw += len(raw); tot_dropped += dropped
        print(f"  raw {len(raw):2d} -> dedup {n_dedup:2d} (-{dropped}) -> fit {len(after):2d} | ctx {chars:5d} chars | {srcs} src | {q[:42]}")
    n = len(QUESTIONS)
    print(f"  TOTALS: avg ctx {tot_chars/n:6.0f} chars/query | avg {tot_pass/n:.1f} passages | {tot_dropped} dups dropped across {n} queries")
    return tot_chars / n

before = run("BEFORE (old: top-5, no dedup, no budget)", 5, False, 10**9)
after  = run("AFTER  (P1: wide rerank + dedup + budget)", 12, True, 6000)
print(f"\n>>> avg context size: {before:.0f} -> {after:.0f} chars/query  ({(1-after/before)*100:+.0f}%)")

# real latency on one query (includes the LLM, so prefill difference shows up)
q = QUESTIONS[0]
for label, rk, dd, bg in [("BEFORE", 5, False, 10**9), ("AFTER", 12, True, 6000)]:
    rag.RANK_K, rag.DEDUP, rag.CTX_BUDGET = rk, dd, bg
    t = time.time(); r = rag.answer(q); dt = time.time() - t
    ctx = sum(len(h["text"]) for h in rag.curate(q))
    print(f"answer() {label}: {dt:5.1f}s | ctx {ctx} chars | {len(r['sources'])} cited | ans {len(r['answer'])} chars")

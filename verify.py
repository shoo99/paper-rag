#!/usr/bin/env python3
"""
verify.py — a claim-verification ("verify") layer for paper-rag.

Inspired by Andrej Karpathy's llm-wiki pattern (the "two-step ingest": first understand,
then surface contradictions). Here we apply the same idea at *answer* time: after the RAG
drafts an answer, decompose it into atomic claims and check each one against the retrieved
sources. Claims a source doesn't actually support are flagged as potential hallucinations.

This is the heart of a research co-scientist: a tool that cites wrong things *confidently*
is worse than no tool. Grounding-in-the-prompt ("answer only from context") is necessary
but not sufficient — models still fabricate. The verify pass is the safety net.

Reuses rag.py's retrieval + local LLM + cross-encoder. Nothing leaves the machine.

Usage:
  RAG_DB=./rag_qdrant_test RAG_EMBED=bge-m3-cpu RAG_LLM=qwen3:8b python verify.py "your question"
"""
import os, json, re, sys, time
import rag

# v2: when strict, a specific number/value/name must appear verbatim in a source (topical
# relevance is NOT enough). v1 (lenient) checked only whether an excerpt was on-topic, which
# let a fabricated statistic ("AUROC 0.804") pass because the cited passage discussed the model.
STRICT = os.environ.get("RAG_VERIFY_STRICT", "0") != "0"


def _json_array(s):
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []


def _json_obj(s):
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def answer_from_hits(question, hits):
    """Same prompt as rag.answer(), but on a fixed hit set so verify checks the *same* sources."""
    ctx = "\n\n".join(f"[{i+1}] ({h['source']} p.{h['page']}) {h['text']}" for i, h in enumerate(hits))
    system = ("You are a research assistant. Answer using ONLY the provided context excerpts. "
              "Cite sources inline as [n]. If the context lacks the answer, say so plainly.")
    return rag.llm(system, f"CONTEXT:\n{ctx}\n\nQUESTION: {question}")


def decompose(answer):
    """Split a draft answer into atomic, independently-checkable factual claims."""
    # an abstention ("the context does not say...") has no factual claims to verify
    if re.search(r"\b(does not|doesn't|do not|no (information|mention)|not (stated|mentioned|found|provided))\b",
                 answer, re.I) and len(answer) < 240:
        return []
    sys = ("Split the text into a JSON array of atomic factual claims. Each claim is ONE "
           "self-contained statement with no pronouns. Drop hedging/meta sentences. "
           "Output ONLY the JSON array of strings.")
    claims = _json_array(rag.llm(sys, answer))
    return [c.strip() for c in claims if isinstance(c, str) and len(c.strip()) > 10][:7]


def check_claim(claim, hits):
    """LLM-as-judge: is `claim` directly supported by the retrieved context?"""
    ctx = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    sys = ('You check whether a CLAIM is DIRECTLY supported by the CONTEXT excerpts. '
           'Reply with strict JSON only: {"supported": true or false, "cite": n or null, "why": "<=8 words"}. '
           'supported=true ONLY if a specific excerpt states it. If the context does not clearly say it, '
           'supported=false even if the claim sounds plausible.')
    if STRICT:
        sys += (' CRITICAL: if the claim contains a specific number, statistic, value, date, or proper '
                'name, that exact value must appear verbatim in an excerpt. A topically-related excerpt '
                'that does NOT contain the value is supported=false. Being on-topic is NOT support.')
    d = _json_obj(rag.llm(sys, f"CONTEXT:\n{ctx}\n\nCLAIM: {claim}"))
    return bool(d.get("supported")), d.get("cite"), str(d.get("why", ""))[:80]


def xenc_support(claim, hits):
    """Cheap secondary signal (reuses the reranker): best claim<->passage relevance."""
    if not rag.HYBRID or not hits:
        return None
    scores = list(rag.reranker().rerank(claim, [h["text"] for h in hits]))
    return round(float(max(scores)), 3) if scores else None


def verify(question):
    """Draft an answer, then verify every claim against the same retrieved sources."""
    hits = rag.curate(question)
    if not hits:
        return {"question": question, "draft": "No matches.", "claims": [],
                "n_claims": 0, "n_grounded": 0, "n_flagged": 0, "grounding_rate": 1.0,
                "flagged_claims": [], "verified_answer": "No matches."}
    draft = answer_from_hits(question, hits)
    claims = decompose(draft)
    results = []
    for c in claims:
        ok, cite, why = check_claim(c, hits)
        results.append({"claim": c, "supported": ok, "cite": cite, "why": why, "xenc": xenc_support(c, hits)})
    flagged = [r for r in results if not r["supported"]]
    # a "verified answer": keep grounded claims, mark unsupported ones instead of silently emitting them
    if not claims:
        verified = draft  # abstention or no extractable claims -> nothing to strip
    else:
        kept = [r["claim"] for r in results if r["supported"]]
        verified = " ".join(kept) if kept else "[verify] No claim in the draft was supported by the retrieved sources."
        if flagged:
            verified += "\n\n⚠️ unsupported (dropped): " + " | ".join(r["claim"] for r in flagged)
    return {"question": question, "draft": draft, "claims": results,
            "n_claims": len(results), "n_grounded": len(results) - len(flagged), "n_flagged": len(flagged),
            "grounding_rate": round((len(results) - len(flagged)) / max(1, len(results)), 2),
            "flagged_claims": [r["claim"] for r in flagged], "verified_answer": verified}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What machine-learning alternative is proposed for predicting drug synergy?"
    t = time.time()
    r = verify(q)
    print(f"\nQ: {q}\n\n--- DRAFT (baseline RAG answer) ---\n{r['draft']}\n")
    print(f"--- VERIFY: {r['n_claims']} claims, {r['n_grounded']} grounded, {r['n_flagged']} flagged "
          f"(grounding {r['grounding_rate']:.0%}) ---")
    for c in r["claims"]:
        mark = "✓" if c["supported"] else "✗ FLAG"
        print(f"  {mark}  {c['claim'][:90]}   [cite {c['cite']}, xenc {c['xenc']}]")
    print(f"\n--- VERIFIED ANSWER ---\n{r['verified_answer']}\n({time.time()-t:.1f}s)")

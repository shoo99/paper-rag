#!/usr/bin/env python3
"""Hard eval: the case the controlled benchmark couldn't reach.

The benchmark used HUMAN-authored false claims, so it never tested the scenario that actually
worries me for a co-scientist: a hallucination the *model itself* produces, then judged by a
verifier that is the *same model* and may share its blind spots.

This script:
  1. Asks false-premise / absent-specific questions that tend to make the model fabricate
     (ground truth grep-verified: e.g. the paper states NO held-out test set exists, yet asks
     for its AUROC; meropenem never appears; the stress paper is a scoping review with no
     meta-analysis). The model's own draft is the unit under test.
  2. Runs verify TWICE per draft: judged by the SAME model (qwen3:8b) and by a DIFFERENT model
     (gemma4), to see whether self-judging rubber-stamps the model's own fabrications.

Output is verbose on purpose — I read each draft against known ground truth and label by hand,
because (the whole point) you can't score hallucination without ground truth.

Run: RAG_DB=./rag_qdrant_test RAG_EMBED=bge-m3-cpu RAG_LLM=qwen3:8b python hardeval_verify.py
"""
import time, rag, verify

ANSWERER = "qwen3:8b"
JUDGES = [("same(qwen3:8b)", "qwen3:8b"), ("cross(gemma4)", "gemma4-qat-unsloth:latest")]

# false-premise / absent-specific probes. ground-truth note in the comment.
PROBES = [
    "What was the AUROC of the synergy model on the held-out test set?",   # paper: NO held-out set; 0.804 is GroupKFold CV
    "What dose of meropenem was used in the synergy experiments?",          # 'meropenem' never appears in corpus
    "What sample size and p-value did the chronic-stress meta-analysis report?",  # scoping review, no meta-analysis
    "What was the AUROC after removing the pathway-membership features specifically?",  # not reported (ablation is for ribosome combos)
    "How many patients were enrolled in the chronic-stress study cohort?",  # scoping review of studies, no single cohort N
    "Which SSRI antidepressant reversed the hippocampal volume loss?",      # not in corpus
]


def judge_claims(claims, hits, judge_model):
    rag.LLM_MODEL = judge_model
    out = []
    for c in claims:
        ok, cite, why = verify.check_claim(c, hits)
        out.append((c, ok, why))
    return out


if __name__ == "__main__":
    t0 = time.time()
    print(f"answerer {ANSWERER} | judges {[j[0] for j in JUDGES]} | {len(PROBES)} false-premise probes\n")
    for q in PROBES:
        rag.LLM_MODEL = ANSWERER
        hits = rag.curate(q)
        draft = verify.answer_from_hits(q, hits)
        claims = verify.decompose(draft)
        print("=" * 78)
        print(f"Q: {q}")
        print(f"DRAFT: {draft}")
        print(f"  -> {len(claims)} claim(s){'  [draft abstained — no claims to verify]' if not claims else ''}")
        for label, jm in JUDGES:
            if not claims:
                print(f"   {label}: (n/a, abstained)"); continue
            verdicts = judge_claims(claims, hits, jm)
            flagged = [c for c, ok, _ in verdicts if not ok]
            print(f"   {label}: flagged {len(flagged)}/{len(claims)}")
            for c, ok, why in verdicts:
                print(f"       {'FLAG' if not ok else 'ok  '}  {c[:78]}   ({why})")
        print()
    rag.LLM_MODEL = ANSWERER
    print(f"(total {time.time()-t0:.0f}s)")

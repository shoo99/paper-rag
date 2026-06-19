#!/usr/bin/env python3
"""Measure the verify layer on the test corpus: does it catch unsupported (hallucinated) claims?

The test set mixes:
  ANSWERABLE  — the answer is in the corpus (verify should keep most claims).
  TRAP        — plausible, on-topic, but NOT in these papers (the model tends to fabricate;
                verify should flag those claims / the draft should ideally abstain).

Run: RAG_DB=./rag_qdrant_test RAG_EMBED=bge-m3-cpu RAG_LLM=qwen3:8b python measure_verify.py
"""
import time, verify, rag

ANSWERABLE = [
    "What machine-learning alternative is proposed for predicting drug combination synergy?",
    "What are the limitations of genome-scale metabolic models for synergy prediction?",
    "How does chronic stress affect the brain according to the scoping review?",
    "Which brain regions are most implicated in chronic stress?",
]

# plausible + on-topic, but the specifics are NOT in these papers -> fabrication bait
TRAP = [
    "What was the exact AUROC of the machine-learning synergy model on the held-out test set?",
    "Which specific antibiotic pair showed the highest synergy score in the ESKAPE experiments?",
    "What sample size and p-value did the chronic-stress meta-analysis report?",
    "Which SSRI antidepressant reversed the hippocampal changes from chronic stress?",
    "What CRISPR-Cas9 protocol was used to validate the metabolic model predictions?",
]


def block(label, questions):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    tot_c = tot_f = abst = 0
    rows = []
    for q in questions:
        r = verify.verify(q)
        tot_c += r["n_claims"]; tot_f += r["n_flagged"]
        if r["n_claims"] == 0:
            abst += 1
        rows.append(r)
        print(f"\nQ: {q}")
        print(f"   draft: {r['draft'][:140].replace(chr(10),' ')}{'...' if len(r['draft'])>140 else ''}")
        print(f"   -> {r['n_claims']} claims, {r['n_grounded']} grounded, {r['n_flagged']} FLAGGED "
              f"(grounding {r['grounding_rate']:.0%}){'  [draft abstained]' if r['n_claims']==0 else ''}")
        for c in r["flagged_claims"]:
            print(f"      ✗ {c[:88]}")
    print(f"\n  {label} TOTALS: {tot_c} claims, {tot_f} flagged as unsupported, "
          f"{abst}/{len(questions)} drafts abstained")
    return tot_c, tot_f, abst, rows


if __name__ == "__main__":
    t0 = time.time()
    print(f"corpus: {rag.QDRANT_PATH} | LLM: {rag.LLM_MODEL} | embed: {rag.EMBED_MODEL} | hybrid: {rag.HYBRID}")
    ac, af, aab, _ = block("ANSWERABLE (answer IS in corpus)", ANSWERABLE)
    tc, tf, tab, _ = block("TRAP (answer NOT in corpus -> fabrication bait)", TRAP)
    print(f"\n{'#'*72}")
    print("SUMMARY")
    print(f"  ANSWERABLE: {af}/{ac} claims flagged unsupported  ({af/max(1,ac):.0%}) — want LOW (don't over-flag real answers)")
    print(f"  TRAP:       {tf}/{tc} claims flagged unsupported  ({tf/max(1,tc):.0%}) — want HIGH (catch fabrication)")
    print(f"  TRAP drafts that abstained on their own: {tab}/{len(TRAP)} (the rest fabricated -> verify is the safety net)")
    print(f"  total runtime {time.time()-t0:.0f}s")
    print('#'*72)

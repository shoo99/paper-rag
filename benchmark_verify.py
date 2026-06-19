#!/usr/bin/env python3
"""Ground-truth benchmark for the verify layer.

The first eval was confounded: 'trap' questions turned out to be answerable (the corpus
*did* contain the AUROC 0.804 etc.), so verify's flags looked like misses/errors when the
real lesson was "I never checked my ground truth." This benchmark fixes that: every claim is
hand-labeled against facts I confirmed by grepping the corpus.

Design — isolate the *verifier* from retrieval:
  Each item = (topic, TRUE claim, FALSE claim). TRUE is a fact present in the corpus; FALSE
  is the same statement with a corrupted number/entity that is confirmed ABSENT. We retrieve
  the supporting context once (using the true claim), then ask the verifier to judge BOTH
  against that *same* context:
    TRUE  -> should be SUPPORTED   (flagging it = false positive)
    FALSE -> should be UNSUPPORTED (passing it = a missed hallucination)
  This measures the verifier's catch-rate (recall) and false-positive rate, v1 vs v2.

Terms were grep-verified: TRUE terms present in corpus; FALSE terms confirmed absent
(avoided cerebellum / crispr / loewe / colistin / 0.50 — those ARE in the corpus).

Run: RAG_DB=./rag_qdrant_test RAG_EMBED=bge-m3-cpu RAG_LLM=qwen3:8b python benchmark_verify.py
"""
import time, rag, verify

# (topic/query, TRUE claim [in corpus], FALSE claim [corrupted, confirmed absent])
ITEMS = [
    ("AUROC of the full-dataset synergy predictor",
     "The full-dataset AUROC of the synergy predictor is 0.804.",
     "The full-dataset AUROC of the synergy predictor is 0.92."),
    ("Random Forest AUC performance",
     "Random Forest achieved an AUC of 0.742.",
     "Random Forest achieved an AUC of 0.95."),
    ("interpretation of an AUROC of 0.627",
     "An AUROC of 0.627 is described as near random.",
     "An AUROC of 0.627 is described as excellent discrimination."),
    ("metabolic features used by the ML pipeline",
     "The metabolic features include pathway membership and gene essentiality scores.",
     "The metabolic features include patient age, BMI, and smoking status."),
    ("brain regions implicated in chronic stress",
     "Chronic stress implicates the hippocampus, amygdala, and prefrontal cortex.",
     "Chronic stress leaves the hippocampus and amygdala unaffected."),
    ("lpxC identity risk score",
     "lpxC has an identity risk score of 0.828.",
     "lpxC has an identity risk score of 0.95."),
    ("how drug synergy was scored",
     "Synergy was scored using a Bliss-type model.",
     "Synergy was scored using the Chou-Talalay combination index."),
    ("methods used in the workflow",
     "An LLM NLP pipeline was part of the workflow.",
     "The top-scoring synergistic pair involved meropenem."),
]


def run(strict):
    verify.STRICT = strict
    caught = fp = 0
    rows = []
    for topic, tc, fc in ITEMS:
        hits = rag.curate(tc)                          # retrieve context that supports the TRUE claim
        t_ok, _, _ = verify.check_claim(tc, hits)      # want True
        f_ok, _, _ = verify.check_claim(fc, hits)      # want False
        if not t_ok:
            fp += 1
        if not f_ok:
            caught += 1
        rows.append((topic, t_ok, f_ok))
    return caught, fp, rows


if __name__ == "__main__":
    t0 = time.time()
    n = len(ITEMS)
    print(f"corpus {rag.QDRANT_PATH} | LLM {rag.LLM_MODEL} | {n} true/false claim pairs\n")
    for strict in (False, True):
        label = "v2 STRICT (numbers/names must match verbatim)" if strict else "v1 lenient (topical support)"
        caught, fp, rows = run(strict)
        print(f"=== {label} ===")
        for topic, t_ok, f_ok in rows:
            tmark = "ok " if t_ok else "FP!"          # true claim should be supported
            fmark = "CAUGHT" if not f_ok else "missed"  # false claim should be flagged
            print(f"   true:{tmark}  false:{fmark:6}  | {topic}")
        print(f"   --> caught {caught}/{n} fabrications (recall {caught/n:.0%}) | "
              f"false-positives {fp}/{n} true claims ({fp/n:.0%})\n")
    print(f"(total {time.time()-t0:.0f}s — controlled ground truth, no retrieval confound)")

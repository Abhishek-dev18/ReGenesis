"""
Measures semantic drift of the agent's DNA relative to generation 0. This is
the direct empirical test of the "value drift across generations" claim
discussed in Section 5.1 of the Theory of Self-Creation paper -- nobody in
the DGM / STOP / Goedel Agent literature measures this directly.

Uses sentence-transformers, which runs on CPU -- no GPU required.
"""
from sentence_transformers import SentenceTransformer, util

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, CPU-friendly
    return _model


def dna_text(dna) -> str:
    return f"{dna.system_prompt} | strategy={dna.decomposition_strategy} | critique={dna.self_critique_enabled}"


def similarity_to_origin(origin_dna, current_dna) -> float:
    model = _get_model()
    emb = model.encode([dna_text(origin_dna), dna_text(current_dna)], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]))

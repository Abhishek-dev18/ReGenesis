"""
Measures semantic similarity of the complete mutable DNA to generation 0.
"""
from sentence_transformers import SentenceTransformer, util

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def dna_text(dna) -> str:
    """Serialize every experimental DNA field that can affect behavior."""
    return (
        f"system_prompt={dna.system_prompt} | "
        f"strategy={dna.decomposition_strategy} | "
        f"temperature={dna.temperature} | "
        f"self_critique={dna.self_critique_enabled} | "
        f"retry_policy={dna.retry_policy}"
    )


def similarity_to_origin(origin_dna, current_dna) -> float:
    model = _get_model()
    emb = model.encode(
        [dna_text(origin_dna), dna_text(current_dna)],
        convert_to_tensor=True,
    )
    return float(util.cos_sim(emb[0], emb[1]))

"""Text cleaning, NER extraction, and entity canonicalization."""
from __future__ import annotations

import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

_KEEP_TYPES = frozenset(["PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "PRODUCT"])


def clean_text(text: str, max_tokens: int = 512) -> str:
    """Strip HTML tags, normalize whitespace, truncate by whitespace-token count."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if len(tokens) > max_tokens:
        text = " ".join(tokens[:max_tokens])
    return text


def extract_entities(
    docs: list[dict],
    model_name: str = "en_core_web_lg",
    batch_size: int = 64,
    keep_types: frozenset[str] = _KEEP_TYPES,
    stop_entity_threshold: float = 0.20,
    canonicalize_threshold: float = 0.85,
) -> dict[int, set[str]]:
    """Return {doc_id -> set of canonical entity strings}.

    Steps:
    1. Run spaCy NER, keep only selected entity types.
    2. Apply stop-entity filter (entities in >threshold% of docs).
    3. Canonicalize near-duplicate entity strings via rapidfuzz clustering.
    """
    import spacy

    nlp = spacy.load(model_name)
    n_docs = len(docs)
    texts = [doc["title"] + " " + doc["body"] for doc in docs]

    raw: dict[int, set[str]] = {}
    for i, doc_nlp in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        raw[i] = {
            ent.text.strip()
            for ent in doc_nlp.ents
            if ent.label_ in keep_types and len(ent.text.strip()) > 1
        }

    # --- Stop-entity filter ---
    entity_doc_count: dict[str, int] = defaultdict(int)
    for entities in raw.values():
        for e in entities:
            entity_doc_count[e] += 1

    threshold_count = stop_entity_threshold * n_docs
    stop_entities = {e for e, c in entity_doc_count.items() if c > threshold_count}
    if stop_entities:
        logger.info("Removing %d stop entities (appear in >%.0f%% of docs)", len(stop_entities), stop_entity_threshold * 100)

    for doc_id in raw:
        raw[doc_id] -= stop_entities

    # --- Canonicalization ---
    all_entities = sorted({e for ents in raw.values() for e in ents})
    canon_map = _build_canon_map(all_entities, canonicalize_threshold)

    result: dict[int, set[str]] = {}
    for doc_id, entities in raw.items():
        result[doc_id] = {canon_map.get(e, e) for e in entities}

    zero_entity_docs = sum(1 for ents in result.values() if not ents)
    logger.info(
        "NER done: %d unique canonical entities; %d/%d docs have 0 entities",
        len({e for ents in result.values() for e in ents}),
        zero_entity_docs,
        n_docs,
    )
    return result


def _build_canon_map(entities: list[str], threshold: float) -> dict[str, str]:
    """Cluster near-duplicate entity strings and map each to its canonical form.

    Uses rapidfuzz token_sort_ratio for robustness to word-order variation
    (e.g., "Tim Cook" vs "Cook, Tim"). Only clusters entities whose lengths
    differ by less than 2x to avoid spurious merges.
    """
    if not entities:
        return {}

    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.warning("rapidfuzz not installed; skipping entity canonicalization")
        return {}

    canon_map: dict[str, str] = {}
    cluster_rep: list[str] = []  # one representative per cluster

    for entity in entities:
        if entity in canon_map:
            continue
        matched = False
        for rep in cluster_rep:
            # Skip pairs with very different lengths (unlikely to be duplicates)
            if max(len(entity), len(rep)) > 2 * min(len(entity), len(rep)):
                continue
            score = fuzz.token_sort_ratio(entity.lower(), rep.lower())
            if score >= threshold * 100:
                # Map entity → rep (prefer the longer/more complete form)
                if len(entity) > len(rep):
                    canon_map[rep] = entity
                    # Update cluster_rep in place
                    idx = cluster_rep.index(rep)
                    cluster_rep[idx] = entity
                else:
                    canon_map[entity] = rep
                matched = True
                break
        if not matched:
            cluster_rep.append(entity)

    return canon_map

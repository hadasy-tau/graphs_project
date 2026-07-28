"""Text cleaning, NER extraction, and entity canonicalization."""
from __future__ import annotations

import logging
import re
import unicodedata
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

def _resolve_canonical(entity: str, canon_map: dict[str, str]) -> str:
    """Follow canonical mapping chain until the final representative."""
    seen: list[str] = []
    current = entity

    while current in canon_map and canon_map[current] != current:
        if current in seen:
            raise ValueError(f"Cycle detected in canonical map for entity: {entity}")
        seen.append(current)
        current = canon_map[current]

    for x in seen:
        canon_map[x] = current

    return current


def extract_entities(
    docs: list[dict],
    model_name: str = "en_core_web_lg",
    batch_size: int = 64,
    keep_types: frozenset[str] = _KEEP_TYPES,
    stop_entity_threshold: float = 0.20,
    canonicalize_threshold: float = 0.90,
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

    # --- Canonicalization diagnostics ---
    resolved_map = {
        e: _resolve_canonical(e, canon_map)
        for e in all_entities
    }

    nontrivial = {
        e: c
        for e, c in resolved_map.items()
        if e != c
    }

    canonical_groups: dict[str, list[str]] = defaultdict(list)
    for e, c in resolved_map.items():
        canonical_groups[c].append(e)

    merged_groups = {
        c: variants
        for c, variants in canonical_groups.items()
        if len(variants) > 1
    }

    logger.info(
        "Canonicalization diagnostics: %d raw entities, %d canonical entities, %d entities remapped, %d merged groups",
        len(all_entities),
        len(set(resolved_map.values())),
        len(nontrivial),
        len(merged_groups),
    )

    for canon, variants in list(merged_groups.items())[:20]:
        logger.info("Canonical group: %r <- %s", canon, variants)

    # --- Apply canonicalization ---
    result: dict[int, set[str]] = {}
    for doc_id, entities in raw.items():
        result[doc_id] = {resolved_map.get(e, e) for e in entities}

    zero_entity_docs = sum(1 for ents in result.values() if not ents)
    logger.info(
        "NER done: %d unique canonical entities; %d/%d docs have 0 entities",
        len({e for ents in result.values() for e in ents}),
        zero_entity_docs,
        n_docs,
    )
    return result




def _entity_norm_key(entity: str) -> str:
    """Return a conservative comparison key for exact deterministic entity merges.

    This is used only for matching, not as the displayed canonical entity.
    It removes purely orthographic differences: casing, accents, spaces,
    punctuation, quote style, and @ handles.
    """
    text = unicodedata.normalize("NFKD", entity)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = "".join(ch for ch in text if ch.isalnum())
    return text


def _choose_entity_rep(variants: list[str]) -> str:
    """Choose a readable representative for a deterministic entity group."""
    return max(
        variants,
        key=lambda s: (
            not s.startswith("@"),   # prefer display names over @handles
            len(s),                  # prefer more complete strings
            s,
        ),
    )

def _build_canon_map(entities: list[str], threshold: float) -> dict[str, str]:
    """Cluster near-duplicate entity strings and map each to its canonical form.

    First performs deterministic exact matching on normalized keys, handling
    safe orthographic variants such as casing, accents, punctuation, spaces,
    and @handles. Then applies rapidfuzz token_sort_ratio as a second pass
    over the remaining representatives.
    """
    if not entities:
        return {}

    canon_map: dict[str, str] = {}

    # --- Pass 1: deterministic exact grouping by normalized key ---
    norm_groups: dict[str, list[str]] = defaultdict(list)
    ungrouped: list[str] = []

    for entity in entities:
        key = _entity_norm_key(entity)

        # Avoid exact-normalizing very short strings, where accidental
        # collisions are more likely.
        if len(key) >= 3:
            norm_groups[key].append(entity)
        else:
            ungrouped.append(entity)

    representatives: list[str] = []

    for variants in norm_groups.values():
        rep = _choose_entity_rep(variants)
        representatives.append(rep)

        for variant in variants:
            if variant != rep:
                canon_map[variant] = rep

    representatives.extend(ungrouped)
    representatives = sorted(set(representatives))

    # --- Pass 2: fuzzy matching over representatives only ---
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.warning("rapidfuzz not installed; skipping fuzzy entity canonicalization")
        return canon_map

    cluster_rep: list[str] = []

    for entity in representatives:
        if entity in canon_map:
            continue

        matched = False
        for rep in cluster_rep:
            # Skip pairs with very different lengths (unlikely to be duplicates)
            if max(len(entity), len(rep)) > 2 * min(len(entity), len(rep)):
                continue

            score = fuzz.token_sort_ratio(entity.lower(), rep.lower())
            if score >= threshold * 100:
                # Map entity → rep, preferring the longer/more complete form
                if len(entity) > len(rep):
                    canon_map[rep] = entity
                    idx = cluster_rep.index(rep)
                    cluster_rep[idx] = entity
                else:
                    canon_map[entity] = rep

                matched = True
                break

        if not matched:
            cluster_rep.append(entity)

    return canon_map

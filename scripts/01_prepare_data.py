"""Phase 1: Download and preprocess MultiHop-RAG; compute embeddings and NER."""
import json
import logging

# Resolves the active config and every output path from the environment, and puts
# the repo root on sys.path. Import before anything from src/.
from experiment import PROCESSED, cfg, ensure_dirs

logger = logging.getLogger(__name__)


def main():
    from src.data.loader import load_multihop_rag, save_jsonl
    from src.data.preprocessor import clean_text, extract_entities
    from src.data.embedder import embed_documents, load_or_compute

    ensure_dirs()

    # --- 1. Load dataset ---
    logger.info("Loading MultiHop-RAG...")
    corpus, queries = load_multihop_rag()

    # --- 2. Clean body text ---
    for doc in corpus:
        doc["body"] = clean_text(doc["body"], max_tokens=cfg["embedding"]["max_tokens"])

    # --- 3. Save corpus ---
    save_jsonl(corpus, PROCESSED / "corpus.jsonl")
    logger.info("Saved %d corpus docs", len(corpus))

    # --- 4. Save queries ---
    save_jsonl(queries, PROCESSED / "queries.jsonl")
    logger.info("Saved %d queries", len(queries))
    _verify_queries(queries)

    # --- 5. Compute and cache document embeddings ---
    emb_path = PROCESSED / "embeddings.pt"
    emb = load_or_compute(
        emb_path,
        embed_documents,
        corpus,
        model_name=cfg["embedding"]["model"],
        batch_size=cfg["embedding"]["batch_size"],
    )
    assert emb.shape == (len(corpus), cfg["embedding"]["dim"]), f"Unexpected shape: {emb.shape}"
    assert not emb.isnan().any(), "NaN values in embeddings"
    logger.info("Embeddings shape: %s", tuple(emb.shape))

    # --- 5b. Compute and cache metadata record embeddings (metadata graph edges) ---
    # PROCESSED is fingerprinted over embedding/ner/metadata_graph, so changing
    # metadata_graph.fields resolves to a different folder and these records are
    # recomputed rather than silently reused. See scripts/experiment.py.
    from src.data.embedder import embed_metadata_records
    meta_emb = load_or_compute(
        PROCESSED / "metadata_embeddings.pt",
        embed_metadata_records,
        corpus,
        fields=cfg["metadata_graph"]["fields"],
        model_name=cfg["embedding"]["model"],
        batch_size=cfg["embedding"]["batch_size"],
    )
    assert meta_emb.shape == (len(corpus), cfg["embedding"]["dim"]), (
        f"Unexpected shape: {meta_emb.shape}"
    )
    logger.info("Metadata record embeddings shape: %s", tuple(meta_emb.shape))

    # --- 6. Compute and cache query embeddings ---
    from src.data.embedder import embed_queries
    q_emb_path = PROCESSED / "query_embeddings.pt"
    load_or_compute(
        q_emb_path,
        embed_queries,
        queries,
        model_name=cfg["embedding"]["model"],
        batch_size=cfg["embedding"]["batch_size"],
        query_prefix=cfg["embedding"].get("query_prefix", ""),
    )

    # --- 7. Extract and cache named entities ---
    entities_path = PROCESSED / "entities.json"
    if entities_path.exists():
        logger.info("Loading cached entities from %s", entities_path)
        with open(entities_path, encoding="utf-8") as f:
            entities_raw = json.load(f)
        entities = {int(k): set(v) for k, v in entities_raw.items()}
    else:
        logger.info("Extracting named entities...")
        ner_cfg = cfg["ner"]
        entities = extract_entities(
            corpus,
            model_name=ner_cfg["model"],
            batch_size=ner_cfg["batch_size"],
            stop_entity_threshold=ner_cfg["stop_entity_threshold"],
            canonicalize_threshold=ner_cfg["canonicalize_threshold"],
        )
        entities_serializable = {k: sorted(v) for k, v in entities.items()}
        with open(entities_path, "w", encoding="utf-8") as f:
            json.dump(entities_serializable, f)
        logger.info("Saved entities for %d docs", len(entities))

    _verify_entities(entities, len(corpus))
    logger.info("Phase 1 complete.")


def _verify_queries(queries):
    no_gold = [q["query_id"] for q in queries if not q["gold_doc_ids"]]
    if no_gold:
        logger.warning("%d queries have no resolved gold_doc_ids", len(no_gold))
    else:
        logger.info("All queries have at least one resolved gold doc.")


def _verify_entities(entities, n_docs):
    zero_entity = sum(1 for ents in entities.values() if not ents)
    frac = zero_entity / n_docs
    if frac > 0.05:
        logger.warning("%.1f%% of docs have 0 entities after NER", frac * 100)
    else:
        logger.info("%.1f%% of docs have 0 entities (acceptable)", frac * 100)


if __name__ == "__main__":
    main()

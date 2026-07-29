"""Phase 2: Build all four graph variants and save them to disk."""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = ROOT / "data" / "processed"
GRAPHS = ROOT / "data" / "graphs"
GRAPHS.mkdir(parents=True, exist_ok=True)
METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "base.yaml") as f:
    cfg = yaml.safe_load(f)


def main():
    from src.data.loader import load_jsonl
    from src.data.embedder import embed_texts
    from src.graph.entity_graph import EntityGraphBuilder
    from src.graph.metadata_graph import MetadataGraphBuilder
    from src.graph.semantic_graph import SemanticGraphBuilder
    from src.graph.combined_graph import CombinedGraphBuilder

    # --- Load preprocessed data ---
    corpus = load_jsonl(PROCESSED / "corpus.jsonl")
    queries = load_jsonl(PROCESSED / "queries.jsonl")
    embeddings = torch.load(PROCESSED / "embeddings.pt", weights_only=True)

    with open(PROCESSED / "entities.json") as f:
        entities = {int(k): set(v) for k, v in json.load(f).items()}

    embed_fn = lambda texts: embed_texts(
        texts,
        model_name=cfg["embedding"]["model"],
        batch_size=cfg["embedding"]["batch_size"],
    )

    # --- Entity graph ---
    shared_knn_k = cfg.get("mutual_knn_k")
    entity_builder = EntityGraphBuilder(
        min_shared_entities=cfg["entity_graph"]["min_shared_entities"],
        mutual_knn_k=shared_knn_k,
    )
    entity_data, entity_tn, entity_te = entity_builder.build(
        corpus, embeddings, entities, edge_embedder=embed_fn
    )
    entity_avg_degree = (entity_data.edge_index.shape[1] // 2) * 2 / len(corpus)
    logger.info("Entity graph avg degree: %.2f", entity_avg_degree)

    # --- Metadata graph ---
    metadata_builder = MetadataGraphBuilder()
    metadata_data, metadata_tn, metadata_te = metadata_builder.build(
        corpus, embeddings, entities, edge_embedder=embed_fn
    )

    # --- Semantic graph ---
    # Shared mutual_knn_k applies to both; otherwise match entity avg degree
    mutual_k = shared_knn_k or _target_knn_k(entity_avg_degree, len(corpus))
    if shared_knn_k is not None:
        logger.info("Semantic graph mutual_knn_k=%d (shared with entity)", mutual_k)
    else:
        logger.info(
            "Semantic graph mutual_knn_k set to %d (targeting entity avg degree %.2f)",
            mutual_k, entity_avg_degree,
        )

    semantic_builder = SemanticGraphBuilder(mutual_knn_k=mutual_k)
    semantic_data, semantic_tn, semantic_te = semantic_builder.build(
        corpus, embeddings, entities, edge_embedder=embed_fn
    )

    # --- Combined graph ---
    combined_builder = CombinedGraphBuilder(
        entity_builder=entity_builder,
        metadata_builder=metadata_builder,
        semantic_builder=semantic_builder,
    )
    combined_data, combined_tn, combined_te = combined_builder.build(
        corpus, embeddings, entities, edge_embedder=embed_fn
    )

    # --- Save graphs ---
    graphs = {
        "entity": (entity_data, entity_tn, entity_te),
        "metadata": (metadata_data, metadata_tn, metadata_te),
        "semantic": (semantic_data, semantic_tn, semantic_te),
        "combined": (combined_data, combined_tn, combined_te),
    }

    builders = {
        "entity": entity_builder,
        "metadata": metadata_builder,
        "semantic": semantic_builder,
        "combined": combined_builder,
    }

    all_stats = []
    for name, (data, tn, te) in graphs.items():
        torch.save(data, GRAPHS / f"{name}_graph.pt")
        tn.to_csv(GRAPHS / f"{name}_textual_nodes.csv", index=False)
        te.to_csv(GRAPHS / f"{name}_textual_edges.csv", index=False)

        # Verify alignment
        assert len(tn) == data.num_nodes, f"{name}: textual_nodes rows != num_nodes"
        assert len(te) == data.edge_index.shape[1], f"{name}: textual_edges rows != num_edges"
        assert not data.edge_attr.isnan().any(), f"{name}: NaN in edge_attr"

        stats = builders[name].compute_stats(data, queries, name=name)
        all_stats.append(stats)
        logger.info("Saved %s graph", name)

    # --- Save graph stats ---
    import csv
    stats_path = METRICS / "graph_stats.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)
    logger.info("Graph stats saved to %s", stats_path)
    logger.info("Phase 2 complete.")


def _target_knn_k(target_avg_degree: float, n_nodes: int) -> int:
    """Estimate mutual k-NN k to approximately match a target average degree.

    For mutual k-NN on N nodes, the expected number of mutual edges is roughly
    k (since an edge is mutual with probability ~k/N). So avg_degree ≈ k.
    """
    k = max(1, round(target_avg_degree))
    return min(k, n_nodes - 1)


if __name__ == "__main__":
    main()

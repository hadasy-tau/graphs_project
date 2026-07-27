"""Abstract base class for document-level graph builders."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import torch
from torch_geometric.data import Data


class GraphBuilder(ABC):
    """Build a document-level graph in G-Retriever-compatible format.

    Each concrete subclass defines a different edge construction strategy
    (entity, metadata, semantic, combined) while sharing the same node set
    (one node per document) and node features (sentence embeddings).
    """

    @abstractmethod
    def get_edges(
        self,
        corpus: list[dict],
        embeddings: torch.Tensor,
        entities: dict[int, set[str]],
    ) -> list[tuple[int, int, str]]:
        """Return list of (src_doc_id, dst_doc_id, edge_text) triples.

        edge_text is a human-readable description of the relationship,
        which will be embedded to produce edge_attr for PCST prize scoring.
        Edges are undirected: include only (i, j) with i < j.
        """

    def build(
        self,
        corpus: list[dict],
        embeddings: torch.Tensor,
        entities: dict[int, set[str]],
        edge_embedder=None,
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 256,
    ) -> tuple[Data, pd.DataFrame, pd.DataFrame]:
        """Build and return (pyg_data, textual_nodes, textual_edges).

        textual_nodes — N rows, columns: node_id, node_attr
        textual_edges — E rows, columns: src, edge_attr, dst

        These DataFrames are passed to retrieval_via_pcst alongside pyg_data.
        """
        n = len(corpus)

        # --- Edges ---
        raw_edges = self.get_edges(corpus, embeddings, entities)
        if not raw_edges:
            # Disconnected graph: no edges; PCST will fall back to node-only retrieval
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, embeddings.shape[1]))
            textual_edges = pd.DataFrame(columns=["src", "edge_attr", "dst"])
        else:
            srcs, dsts, edge_texts = zip(*raw_edges)
            # Make undirected by adding reverse edges
            all_srcs = list(srcs) + list(dsts)
            all_dsts = list(dsts) + list(srcs)
            all_texts = list(edge_texts) * 2

            edge_index = torch.tensor([all_srcs, all_dsts], dtype=torch.long)

            # Embed edge texts (same model as nodes → same vector space)
            if edge_embedder is not None:
                edge_embs = edge_embedder(all_texts)
            else:
                from src.data.embedder import embed_texts
                edge_embs = embed_texts(all_texts, model_name=model_name, batch_size=batch_size)
            edge_attr = edge_embs.float()

            textual_edges = pd.DataFrame({
                "src": [corpus[i]["title"] for i in all_srcs],
                "edge_attr": all_texts,
                "dst": [corpus[i]["title"] for i in all_dsts],
            })

        # --- Nodes ---
        textual_nodes = pd.DataFrame({
            "node_id": [doc["doc_id"] for doc in corpus],
            "node_attr": [doc["title"] + ". " + doc["body"][:300] for doc in corpus],
        })

        data = Data(
            x=embeddings.float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_idx=list(range(n)),
            edge_idx=list(range(len(textual_edges))),
        )

        return data, textual_nodes, textual_edges

    # --- Graph statistics ---

    def compute_stats(
        self,
        data: Data,
        queries: list[dict],
        name: str = "graph",
    ) -> dict:
        """Compute and log key graph statistics including oracle connectivity."""
        import networkx as nx

        n = data.num_nodes
        e = data.edge_index.shape[1] // 2  # undirected edge count

        g = nx.Graph()
        g.add_nodes_from(range(n))
        if e > 0:
            edges = data.edge_index.t().tolist()
            # deduplicate since we have both directions
            seen = set()
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    g.add_edge(u, v)

        avg_degree = (2 * g.number_of_edges()) / n if n > 0 else 0
        n_components = nx.number_connected_components(g)
        largest_cc = max(len(c) for c in nx.connected_components(g)) if n > 0 else 0

        # Oracle connectivity: fraction of gold doc pairs directly connected
        connected_pairs = 0
        total_pairs = 0
        for q in queries:
            gold = q["gold_doc_ids"]
            for i in range(len(gold)):
                for j in range(i + 1, len(gold)):
                    total_pairs += 1
                    if g.has_edge(gold[i], gold[j]):
                        connected_pairs += 1

        oracle_conn = connected_pairs / total_pairs if total_pairs > 0 else 0.0

        stats = {
            "name": name,
            "n_nodes": n,
            "n_edges": e,
            "avg_degree": round(avg_degree, 3),
            "n_components": n_components,
            "largest_cc": largest_cc,
            "oracle_connectivity": round(oracle_conn, 4),
        }

        import logging
        logging.getLogger(__name__).info(
            "[%s] nodes=%d edges=%d avg_deg=%.2f components=%d oracle_conn=%.3f",
            name, n, e, avg_degree, n_components, oracle_conn,
        )
        return stats

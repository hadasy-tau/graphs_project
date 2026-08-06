"""Regenerate the paper's figures from the committed experiment_handoff CSVs.

    python generate_figures.py

No Colab/Drive dependency: reads from experiment_handoff/ and writes to figures/
under the repo root, so the figures can be reproduced from a plain clone.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = Path(__file__).resolve().parent
HANDOFF = ROOT / "experiment_handoff"
SAVE_DIR = ROOT / "figures"
SAVE_DIR.mkdir(exist_ok=True)

# ACL-compliant styling configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'text.usetex': False  # Set to True if local LaTeX is installed
})

# Color palette & markers for consistency across figures
COLORS = {
    'combined': '#2ca02c',  # Green
    'semantic': '#1f77b4',  # Blue
    'entity':   '#ff7f0e',  # Orange
    'metadata': '#d62728'   # Red
}

MARKERS = {
    'combined': 'o',
    'semantic': 's',
    'entity':   '^',
    'metadata': 'D'
}


def _save(fig, name):
    fig.savefig(SAVE_DIR / f"{name}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(SAVE_DIR / f"{name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 1: PCST Sensitivity (Precision & Recall vs. Avg Retrieved Size)
# -----------------------------------------------------------------------------
def generate_figure_1():
    # k=17 PCST sensitivity sweep (topk/cost_e variants for each graph type)
    df = pd.read_csv(HANDOFF / "analysis" / "pcst_sensitivity_k17_gold_dedup.csv")
    df = df.rename(columns={"graph_type": "graph", "avg_retrieved_size": "size",
                            "evidence_recall": "recall"})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)

    for g_type, df_g in df.groupby('graph'):
        df_g = df_g.sort_values('size')
        label_name = g_type.capitalize()
        ax1.plot(df_g['size'], df_g['precision'], label=label_name,
                 color=COLORS[g_type], marker=MARKERS[g_type], linewidth=1.8, markersize=5.5)
        ax2.plot(df_g['size'], df_g['recall'], label=label_name,
                 color=COLORS[g_type], marker=MARKERS[g_type], linewidth=1.8, markersize=5.5)

    ax1.set_title('(a) Precision vs. Retrieved Set Size')
    ax1.set_xlabel('Average Retrieved Documents')
    ax1.set_ylabel('Precision')
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.set_title('(b) Evidence Recall vs. Retrieved Set Size')
    ax2.set_xlabel('Average Retrieved Documents')
    ax2.set_ylabel('Evidence Recall')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(title='Graph Type', frameon=True, loc='lower right')

    fig.tight_layout()
    _save(fig, 'fig_pcst_sensitivity')


# -----------------------------------------------------------------------------
# Figure 2: Oracle Connectivity vs. Matched-Size Recall (Size = 10)
# -----------------------------------------------------------------------------
def generate_figure_2():
    stats = pd.read_csv(HANDOFF / "arms" / "mutual_knn_k17" / "metrics" / "graph_stats.csv")
    oracle_conn = dict(zip(stats["name"], stats["oracle_connectivity"]))

    interp = pd.read_csv(
        HANDOFF / "analysis" / "pcst_sensitivity_interpolated_by_size_gold_dedup.csv"
    )
    row = interp[(interp["metric"] == "evidence_recall") & (interp["size"] == 10.0)].iloc[0]
    graphs = ['metadata', 'entity', 'semantic', 'combined']
    recall_size10 = {g: row[g] for g in graphs}

    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    x_vals = [oracle_conn[g] for g in graphs]
    y_vals = [recall_size10[g] for g in graphs]

    ax.plot(x_vals, y_vals, color='gray', linestyle='--', alpha=0.6, zorder=2)

    for g in graphs:
        ax.scatter(oracle_conn[g], recall_size10[g], color=COLORS[g],
                   marker=MARKERS[g], s=80, label=g.capitalize(), zorder=5)

    # Offset annotations for visual clarity
    offsets = {
        'metadata': (-15, -12),
        'entity':   (15, -8),
        'semantic': (-22, 8),
        'combined': (-25, 8)
    }

    for g in graphs:
        ax.annotate(g.capitalize(), (oracle_conn[g], recall_size10[g]),
                    textcoords="offset points", xytext=offsets[g], ha='center', fontweight='bold', fontsize=9)

    ax.set_title('Oracle Connectivity vs. Recall at Size 10')
    ax.set_xlabel('Oracle Connectivity (Direct Gold Edge Ratio)')
    ax.set_ylabel('Interpolated Recall at Size = 10')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0.43, 0.72)
    ax.set_ylim(0.675, 0.720)

    fig.tight_layout()
    _save(fig, 'fig_oracle_vs_recall')


# -----------------------------------------------------------------------------
# Figure 3: Evidence Recall by Question Type (Inference, Comparison, Temporal)
# -----------------------------------------------------------------------------
def generate_figure_3():
    qtype = pd.read_csv(HANDOFF / "arms" / "mutual_knn_k17" / "metrics" / "by_qtype_table.csv")
    pcst_conditions = {f"{g}_pcst" for g in ('entity', 'metadata', 'semantic', 'combined')}
    qtype = qtype[qtype["condition"].isin(pcst_conditions)].copy()

    qtype_label = {"inference_query": "Inference", "comparison_query": "Comparison",
                   "temporal_query": "Temporal"}
    graph_label = {"entity_pcst": "Entity", "metadata_pcst": "Metadata",
                   "semantic_pcst": "Semantic", "combined_pcst": "Combined"}

    df_qtype = pd.DataFrame({
        "Graph": qtype["condition"].map(graph_label),
        "Question Type": pd.Categorical(qtype["question_type"].map(qtype_label),
                                        categories=["Inference", "Comparison", "Temporal"],
                                        ordered=True),
        "Recall": qtype["evidence_recall"],
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    sns.barplot(data=df_qtype, x='Question Type', y='Recall', hue='Graph',
                hue_order=['Entity', 'Metadata', 'Semantic', 'Combined'],
                palette={'Entity': COLORS['entity'], 'Metadata': COLORS['metadata'],
                         'Semantic': COLORS['semantic'], 'Combined': COLORS['combined']}, ax=ax)

    ax.set_title('Evidence Recall by Question Type ($k=17$ PCST)')
    ax.set_ylabel('Evidence Recall')
    ax.set_ylim(0.55, 0.85)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(title='Graph Construction', frameon=True, loc='upper left')

    fig.tight_layout()
    _save(fig, 'fig_qtype_breakdown')


if __name__ == '__main__':
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    print(f"Figures successfully generated and saved to: {SAVE_DIR}")

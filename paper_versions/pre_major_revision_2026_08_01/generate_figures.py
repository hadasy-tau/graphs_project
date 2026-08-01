import os
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Mount Google Drive
drive.mount('/content/drive')

# Define the save directory and ensure it exists
SAVE_DIR = '/content/drive/MyDrive/graphrag_mknn_2026_07_30/figures/'
os.makedirs(SAVE_DIR, exist_ok=True)

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

# -----------------------------------------------------------------------------
# Figure 1: PCST Sensitivity (Precision & Recall vs. Avg Retrieved Size)
# -----------------------------------------------------------------------------
def generate_figure_1():
    # Experimental data points from k=17 PCST sensitivity sweep
    data = [
        {'graph': 'combined', 'size': 7.89,  'precision': 0.2109, 'recall': 0.6305},
        {'graph': 'combined', 'size': 9.40,  'precision': 0.1984, 'recall': 0.7027},
        {'graph': 'combined', 'size': 9.89,  'precision': 0.1909, 'recall': 0.7099},
        {'graph': 'combined', 'size': 9.97,  'precision': 0.1902, 'recall': 0.7108},
        {'graph': 'combined', 'size': 12.27, 'precision': 0.1657, 'recall': 0.7647},
        
        {'graph': 'entity',   'size': 8.68,  'precision': 0.1969, 'recall': 0.5895},
        {'graph': 'entity',   'size': 9.89,  'precision': 0.1925, 'recall': 0.6843},
        {'graph': 'entity',   'size': 11.75, 'precision': 0.1699, 'recall': 0.7047},
        {'graph': 'entity',   'size': 12.96, 'precision': 0.1579, 'recall': 0.7170},
        {'graph': 'entity',   'size': 14.97, 'precision': 0.1435, 'recall': 0.7626},
        
        {'graph': 'metadata', 'size': 7.24,  'precision': 0.2346, 'recall': 0.5983},
        {'graph': 'metadata', 'size': 8.67,  'precision': 0.2131, 'recall': 0.6641},
        {'graph': 'metadata', 'size': 9.71,  'precision': 0.1966, 'recall': 0.6794},
        {'graph': 'metadata', 'size': 10.28, 'precision': 0.1881, 'recall': 0.6870},
        {'graph': 'metadata', 'size': 12.57, 'precision': 0.1622, 'recall': 0.7339},
        
        {'graph': 'semantic', 'size': 6.91,  'precision': 0.2364, 'recall': 0.6013},
        {'graph': 'semantic', 'size': 8.35,  'precision': 0.2181, 'recall': 0.6770},
        {'graph': 'semantic', 'size': 9.18,  'precision': 0.2036, 'recall': 0.6897},
        {'graph': 'semantic', 'size': 9.70,  'precision': 0.1966, 'recall': 0.7003},
        {'graph': 'semantic', 'size': 11.89, 'precision': 0.1705, 'recall': 0.7511},
    ]
    df = pd.DataFrame(data)

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

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig_pcst_sensitivity.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(SAVE_DIR, 'fig_pcst_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Figure 2: Oracle Connectivity vs. Matched-Size Recall (Size = 10)
# -----------------------------------------------------------------------------
def generate_figure_2():
    oracle_conn = {'metadata': 0.4576, 'entity': 0.4869, 'semantic': 0.5441, 'combined': 0.6951}
    recall_size10 = {'metadata': 0.6833, 'entity': 0.6855, 'semantic': 0.7073, 'combined': 0.7115}

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    
    graphs = ['metadata', 'entity', 'semantic', 'combined']
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

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig_oracle_vs_recall.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(SAVE_DIR, 'fig_oracle_vs_recall.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Figure 3: Evidence Recall by Question Type (Inference, Comparison, Temporal)
# -----------------------------------------------------------------------------
def generate_figure_3():
    qtype_data = {
        'Graph': ['Entity', 'Entity', 'Entity', 
                  'Metadata', 'Metadata', 'Metadata',
                  'Semantic', 'Semantic', 'Semantic', 
                  'Combined', 'Combined', 'Combined'],
        'Question Type': ['Inference', 'Comparison', 'Temporal'] * 4,
        'Recall': [0.637, 0.719, 0.780, 
                   0.641, 0.682, 0.730, 
                   0.620, 0.707, 0.762, 
                   0.635, 0.734, 0.780]
    }
    df_qtype = pd.DataFrame(qtype_data)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    sns.barplot(data=df_qtype, x='Question Type', y='Recall', hue='Graph', 
                palette={'Entity': COLORS['entity'], 'Metadata': COLORS['metadata'], 
                         'Semantic': COLORS['semantic'], 'Combined': COLORS['combined']}, ax=ax)

    ax.set_title('Evidence Recall by Question Type ($k=17$ PCST)')
    ax.set_ylabel('Evidence Recall')
    ax.set_ylim(0.55, 0.85)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(title='Graph Construction', frameon=True, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fig_qtype_breakdown.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(SAVE_DIR, 'fig_qtype_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    print(f"Figures successfully generated and saved to: {SAVE_DIR}")
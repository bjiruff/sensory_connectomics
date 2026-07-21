import sys
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import FIGS_DIR
from connectomics.utils.extract_sensory import _get_sensory
from connectomics.modeling.diffusion_models import sensory_orders
from connectomics.plotting.order_stats import synapse_flow_per_order

max_order = 6 # Plot up to and including this order.

COLORS = {
    "feedforward": "#1F75FE",
    "lateral":     "#9E9E9E",
    "feedback":    "#FF2800",
}
LEGEND_ORDER = ["feedforward", "lateral", "feedback"]

def synapse_flow_by_modality(dfs_by_modality, order_num):
    """
    dfs_by_modality: {modality: synapse_flow_per_order(...) result}

    Returns a DataFrame with columns: modality, feedforward, lateral, feedback,
    with one row per modality that has data for order_num.
    """
    rows = []
    for modality, df in dfs_by_modality.items():
        match = df[df["order"] == order_num]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append({
            "modality": modality,
            "feedforward": row["feedforward"],
            "lateral": row["lateral"],
            "feedback": row["feedback"],
        })
    return pd.DataFrame(rows)

def plot_synapse_flow_per_modality(df, order_num, ax=None, show_legend=True):
    if df.empty:
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    x = list(range(len(df)))
    bottom = np.zeros(len(df))
    for category in ["feedback", "lateral", "feedforward"]:
        ax.bar(x, df[category].values, bottom=bottom, color=COLORS[category], label=category)
        bottom += df[category].values

    ax.set_xticks(x)
    ax.set_xticklabels(df["modality"].tolist(), rotation=45, ha="right")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Proportion of output synapses")
    ax.set_title(f"Synapse flow by modality at order {order_num}")
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        order_idx = [labels.index(cat) for cat in LEGEND_ORDER]
        ax.legend([handles[i] for i in order_idx], [labels[i] for i in order_idx])
    ax.set_ylim(0, 1)
    return ax

def main(dataset="fafb", mode="ids"):
    sensory_ids, sensory_types = _get_sensory(dataset)

    if mode == "ids":
        connections_df = DATASETS[dataset].cell_connections_df
        sensory = sensory_ids
        edges = [
            (row.pre_root_id, row.post_root_id, {"weight": row.syn_count})
            for row in connections_df.itertuples()
        ]
    elif mode == "types":
        connections_df = DATASETS[dataset].type_connections_df
        sensory = sensory_types
        edges = [
            (row.pre_type, row.post_type, {"weight": row.syn_count})
            for row in connections_df.itertuples()
            if row.pre_type != row.post_type
        ]
    G = nx.DiGraph()
    G.add_edges_from(edges)

    orders = sensory_orders(G, sensory)

    dfs_by_modality = {
        modality: synapse_flow_per_order(G, orders, modality, max_order)
        for modality in sensory.keys()
    }

    order_nums = list(range(1, max_order + 1))
    ncols = min(3, len(order_nums))
    nrows = math.ceil(len(order_nums) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, order_num in zip(axes, order_nums):
        df = synapse_flow_by_modality(dfs_by_modality, order_num)
        plot_synapse_flow_per_modality(df, order_num, ax=ax, show_legend=False)
    for ax in axes[len(order_nums):]:
        ax.set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[cat]) for cat in LEGEND_ORDER]
    fig.legend(handles, LEGEND_ORDER, loc="lower center", ncol=len(LEGEND_ORDER))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(str(FIGS_DIR / f"{dataset}_synapse_flow_per_modality.png"), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fafb"
    main(dataset)

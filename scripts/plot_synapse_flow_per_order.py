import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import FIGS_DIR
from connectomics.utils.extract_sensory import _get_sensory
from connectomics.modeling.diffusion_models import sensory_orders
from connectomics.plotting.order_stats import synapse_flow_per_order

max_order = 7 # Plot up to an including this order.

def plot_synapse_flow_per_order(G, orders, modality, max_order, ax=None):
    df = synapse_flow_per_order(G, orders, modality, max_order)

    if df.empty:
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "feedforward": "#2196F3",
        "lateral":     "#9E9E9E",
        "feedback":    "#F44336",
    }

    x = list(range(len(df)))
    bottom = np.zeros(len(df))
    for category in ["feedforward", "lateral", "feedback"]:
        ax.bar(x, df[category].values, bottom=bottom, color=colors[category], label=category)
        bottom += df[category].values

    ax.set_xticks(x)
    ax.set_xticklabels(df["order"].tolist())
    ax.set_xlabel("Order")
    ax.set_ylabel("Proportion of output synapses")
    ax.set_title(f"Synapse flow by order relative to {modality} layer")
    ax.legend()
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

    for modality in sensory.keys():
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_synapse_flow_per_order(G, orders, modality, max_order=max_order, ax=ax)
        plt.tight_layout()
        plt.savefig(str(FIGS_DIR / f"{dataset}_synapse_flow_{modality}.png"), dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fafb"
    main(dataset)
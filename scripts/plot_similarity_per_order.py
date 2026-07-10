import sys
import networkx as nx
import matplotlib.pyplot as plt
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import FIGS_DIR
from connectomics.utils.extract_sensory import _get_sensory
from connectomics.modeling.diffusion_models import sensory_orders
from connectomics.plotting.order_stats import similarity_per_order


def plot_similarity_per_order(orders, order_num, ax=None):
    df = similarity_per_order(orders, order_num)

    if df.empty:
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    matrix = df.values
    modalities = df.index.tolist()
    n = len(modalities)

    cmap = plt.get_cmap("Blues")
    norm_obj = plt.Normalize(vmin=0.0, vmax=1.0)

    ax.imshow(matrix, cmap=cmap, norm=norm_obj)

    fontsize = max(5, min(10, 100 // n))
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            rgba = cmap(norm_obj(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "black" if lum > 0.5 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=fontsize)

    ax.set_xticks(range(n))
    ax.set_xticklabels(modalities, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(modalities)
    ax.set_title(f"Neuron similarity at order {order_num} relative to sensory layer")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Jaccard index")

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

    for order_num in [2, 3, 4, 5]:
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_similarity_per_order(orders, order_num, ax=ax)
        plt.tight_layout()
        plt.savefig(str(FIGS_DIR / f"{dataset}_order_{order_num}_similarity.png"), dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fafb"
    main(dataset)
import sys
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from connectomics.processing.data_loader import DATASETS
from connectomics.config.consts import MODALITY_COLORS
from connectomics.config.paths import FIGS_DIR
from connectomics.plotting.order_stats import cells_per_order
from connectomics.utils.extract_sensory import _get_sensory
from connectomics.modeling.diffusion_models import sensory_orders

def plot_cells_per_order(G, orders, ax=None):
    df = cells_per_order(G, orders)
    palette = {m: MODALITY_COLORS[m] for m in df["modality"].unique()}

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=df, x="order", y="count", hue="modality", palette=palette, marker="o", ax=ax)
    ax.set_xticks(sorted(df["order"].unique()))
    ax.set_xlabel("Order Relative to Sensory Layer")
    ax.set_ylabel("Number of Cells")
    ax.set_title("Cells per Order by Modality")
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

    ax = plot_cells_per_order(G, orders)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / f"{dataset}_cells_per_order.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fafb"
    main(dataset)
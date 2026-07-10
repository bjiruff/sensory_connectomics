import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import FIGS_DIR
from connectomics.utils.extract_sensory import _get_sensory
from connectomics.modeling.diffusion_models import sensory_orders
from connectomics.plotting.order_stats import cells_per_class_within_order


def plot_cells_per_class_within_order(orders, order_num, classification, dataset, mode, ax=None):
    df = cells_per_class_within_order(orders, order_num, classification, dataset, mode)

    if df.empty:
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    data_mtrx = df.values.astype(float)
    log_mtrx = np.log1p(data_mtrx)
    rows = df.index.tolist()
    cols = df.columns.tolist()
    totals = data_mtrx.sum(axis=1).astype(int)
    n_rows, n_cols = data_mtrx.shape

    cmap = plt.get_cmap("Greens")
    norm_obj = plt.Normalize(vmin=log_mtrx.min(), vmax=log_mtrx.max())

    ax.imshow(log_mtrx, cmap=cmap, norm=norm_obj)
    ax.set_xlim(-1.5, n_cols - 0.5)

    # Gray uncolored column for totals. Outside the colormap so it doesn't affect scaling.
    ax.axvspan(-1.5, -0.5, color="#D0D0D0", zorder=2)

    # Font size scaled to fit cells: smaller grids get larger text.
    fontsize = max(5, min(10, 100 // max(n_cols + 1, n_rows)))

    for i in range(n_rows):
        for j in range(n_cols):
            if data_mtrx[i, j] == 0:
                continue
            rgba = cmap(norm_obj(log_mtrx[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "black" if lum > 0.5 else "white"
            ax.text(j, i, int(data_mtrx[i, j]), ha="center", va="center", color=color, fontsize=fontsize, zorder=3)

    for i in range(n_rows):
        ax.text(-1, i, str(totals[i]), ha="center", va="center", color="black", fontsize=fontsize, zorder=3)

    ax.set_xticks([-1] + list(range(n_cols)))
    ax.set_xticklabels(["total"] + cols, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows)

    ax.set_title(f"Number of cells per class in order {order_num}")
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
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_cells_per_class_within_order(orders, order_num, "class", dataset, "id", ax=ax)
        plt.tight_layout()
        plt.savefig(str(FIGS_DIR / f"{dataset}_order_{order_num}_classes.png"), dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fafb"
    main(dataset)

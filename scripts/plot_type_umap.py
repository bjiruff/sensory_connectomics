import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import sparse
from umap import UMAP
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import FIGS_DIR


def _vectorize_types(type_connections_df):
    """
    Represent each cell type as the concatenation of its outgoing synapse-count
    profile (as pre_type, to every other type) and incoming synapse-count profile
    (as post_type, from every other type).

    type_connections_df only lists a small fraction of all possible type pairs
    (e.g. ~0.6% for fafb), so a dense (n_types, n_types) matrix wastes gigabytes
    of memory on zeros. Build sparse matrices instead.

    Returns (types, features) where types is a sorted Index of type names and
    features is a sparse (n_types, 2 * n_types) CSR matrix aligned with it.
    """
    types = pd.Index(
        sorted(set(type_connections_df["pre_type"]) | set(type_connections_df["post_type"]))
    )
    type_to_idx = {t: i for i, t in enumerate(types)}
    n = len(types)

    pre_idx = type_connections_df["pre_type"].map(type_to_idx).to_numpy()
    post_idx = type_connections_df["post_type"].map(type_to_idx).to_numpy()
    weights = type_connections_df["syn_count"].to_numpy()

    out_counts = sparse.coo_matrix((weights, (pre_idx, post_idx)), shape=(n, n)).tocsr()
    in_counts = sparse.coo_matrix((weights, (post_idx, pre_idx)), shape=(n, n)).tocsr()

    features = sparse.hstack([out_counts, in_counts], format="csr")
    return types, features


def _get_type_colors(dataset, types):
    # Color each type by its most common super_class, for interpretability of clusters.
    celltypes_df = DATASETS[dataset].celltypes_df
    classifications_df = DATASETS[dataset].classifications_df

    id_to_type = celltypes_df.set_index("root_id")["primary_type"]
    type_super_class = (
        classifications_df.assign(primary_type=classifications_df["root_id"].map(id_to_type))
        .dropna(subset=["primary_type"])
        .groupby("primary_type")["super_class"]
        .agg(lambda s: s.value_counts().idxmax() if not s.empty else "Unknown")
    )

    super_classes = type_super_class.reindex(types).fillna("Unknown")
    unique_classes = sorted(sc for sc in super_classes.unique() if sc != "Unknown")

    cmap = plt.colormaps["tab20"].resampled(max(len(unique_classes), 1))
    class_colormap = {sc: cmap(i) for i, sc in enumerate(unique_classes)}
    class_colormap["Unknown"] = (0.6, 0.6, 0.6, 1.0)

    colors = [class_colormap[sc] for sc in super_classes]
    return colors, class_colormap


def plot_type_umap(dataset="fafb", random_state=42, ax=None):
    type_connections_df = DATASETS[dataset].type_connections_df
    types, features = _vectorize_types(type_connections_df)

    reducer = UMAP(n_components=2, metric="cosine", random_state=random_state)
    embedding = reducer.fit_transform(features)

    colors, class_colormap = _get_type_colors(dataset, types)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=12, alpha=0.8, linewidths=0)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP of cell-type connectivity profiles ({dataset})")

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", label=sc, markerfacecolor=color, markersize=8)
        for sc, color in class_colormap.items()
    ]
    ax.legend(handles=legend_elements, title="Super class", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    return ax, types, embedding


def main(dataset="fafb"):
    ax, types, embedding = plot_type_umap(dataset=dataset)
    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / f"{dataset}_type_umap.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fafb"
    main(dataset)

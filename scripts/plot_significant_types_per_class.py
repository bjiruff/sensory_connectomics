import sys
import pickle
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import NULL_CONNECTOMES_DIR, FIGS_DIR
from connectomics.config.consts import MODALITIES
from connectomics.modeling.significance import empirical_pvalue, benjamini_hochberg

dataset = "fafb"       # Specify which dataset to read from.
classification = "class"  # Column of classifications_df to group cell types by.
alpha = 0.05             # FDR significance threshold.


def significant_types_per_class(actual, nulls, dataset, classification="class", modalities=MODALITIES, alpha=0.05):
    """
    For each class annotation, counts how many of its cell types show a significant
    weighted sensory-input proportion for each modality.

    Significance is assessed per cell type with a one-tailed empirical p-value per
    modality (proportion of null replicates >= the observed value), then
    Benjamini-Hochberg FDR-corrected across the `modalities` tested for that cell type
    (n=len(modalities)), matching plot_sensory_null_comparison.py.

    Each cell type's class is looked up from a single representative cell (its root_id
    in celltypes_df), under the assumption that all cells sharing a primary_type share
    a class. Classes belonging to the sensory neurons themselves (super_class ==
    "sensory") are excluded, since this plot is about downstream sensory input, not
    the sensory neurons that are the source of it.

    actual: {cell_type: {modality: value, ..., "total": value}}, e.g. loaded from
            {dataset}_actual_weighted_results.pkl.
    nulls: {cell_type: {modality: np.ndarray of null values, ..., "total": ndarray}},
           e.g. loaded from {dataset}_null_weighted_results.pkl.

    Returns (sig_counts, totals): sig_counts is a DataFrame (classes x modalities) of
    the number of significant cell types per class/modality; totals is a Series
    (indexed by class) of the total number of cell types tested in that class.
    """
    celltypes_df = DATASETS[dataset].celltypes_df
    classifications_df = DATASETS[dataset].classifications_df

    if classification not in classifications_df:
        raise ValueError("classification does not exist in classifications_df")

    sensory_classes = set(
        classifications_df.loc[classifications_df["super_class"] == "sensory", classification].unique()
    )

    root_by_type = celltypes_df.drop_duplicates("primary_type").set_index("primary_type")["root_id"]
    class_by_root = classifications_df.set_index("root_id")[classification]

    sig_counts = defaultdict(lambda: {mod: 0 for mod in modalities})
    totals = defaultdict(int)

    for cell_type in actual:
        if cell_type not in root_by_type.index:
            print(f"Warning: '{cell_type}' does not exist in dataset.")
            continue

        root_id = root_by_type[cell_type]
        class_ = class_by_root.get(root_id)
        if class_ is None or pd.isna(class_):
            class_ = "unknown_"
        if class_ in sensory_classes:
            continue

        pvalues = [empirical_pvalue(actual[cell_type][mod], nulls[cell_type][mod]) for mod in modalities]
        qvalues = benjamini_hochberg(pvalues)

        totals[class_] += 1
        for mod, q in zip(modalities, qvalues):
            if q < alpha:
                sig_counts[class_][mod] += 1

    classes = sorted(totals.keys())
    sig_df = pd.DataFrame([sig_counts[c] for c in classes], index=classes)[modalities]
    totals_series = pd.Series(totals)[classes]
    return sig_df, totals_series


def plot_significant_types_per_class(sig_df, totals, ax=None):
    if sig_df.empty:
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    # Transpose for display: rows = modalities, columns = classes.
    sig_mtrx = sig_df.T
    proportions = sig_mtrx.div(totals, axis=1).values
    rows = [mod.capitalize() for mod in sig_mtrx.index.tolist()]
    cols = sig_mtrx.columns.tolist()
    row_totals = sig_mtrx.sum(axis=1).astype(int)
    n_rows, n_cols = proportions.shape

    cmap = plt.get_cmap("Reds")
    norm_obj = plt.Normalize(vmin=0.0, vmax=1.0)

    ax.imshow(proportions, cmap=cmap, norm=norm_obj)
    ax.set_xlim(-1.5, n_cols - 0.5)

    # Gray uncolored column for totals. Outside the colormap so it doesn't affect scaling.
    ax.axvspan(-1.5, -0.5, color="#D0D0D0", zorder=2)

    # Font size scaled to fit cells: smaller grids get larger text.
    fontsize = max(5, min(10, 100 // max(n_cols + 1, n_rows)))

    for i in range(n_rows):
        for j in range(n_cols):
            if sig_mtrx.values[i, j] == 0:
                continue
            rgba = cmap(norm_obj(proportions[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "black" if lum > 0.5 else "white"
            ax.text(j, i, int(sig_mtrx.values[i, j]), ha="center", va="center", color=color, fontsize=fontsize, zorder=3)

    for i in range(n_rows):
        ax.text(-1, i, str(int(row_totals.iloc[i])), ha="center", va="center", color="black", fontsize=fontsize, zorder=3)

    ax.set_xticks([-1] + list(range(n_cols)))
    ax.set_xticklabels(["total"] + cols, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows)

    ax.set_title("Cell types with significant sensory input per class")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Proportion of cell types significant")

    return ax


def main(dataset="fafb", classification="class", alpha=0.05):
    actual_path = NULL_CONNECTOMES_DIR / f"{dataset}_actual_weighted_results.pkl"
    null_path = NULL_CONNECTOMES_DIR / f"{dataset}_null_weighted_results.pkl"

    with open(str(actual_path), "rb") as file:
        actual = pickle.load(file)
    with open(str(null_path), "rb") as file:
        nulls = pickle.load(file)

    sig_df, totals = significant_types_per_class(actual, nulls, dataset, classification=classification, alpha=alpha)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_significant_types_per_class(sig_df, totals, ax=ax)
    fig.tight_layout()

    out_path = FIGS_DIR / f"{dataset}_significant_types_per_class.png"
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fafb"
    main(dataset, classification, alpha)

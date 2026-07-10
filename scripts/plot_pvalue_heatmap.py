import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from connectomics.config.paths import NULL_CONNECTOMES_DIR, FIGS_DIR
from connectomics.config.consts import MODALITIES
from connectomics.modeling.significance import empirical_pvalue, benjamini_hochberg

# Specify which cell types to plot (rows of the heatmap).
CELL_TYPES = ["OA-ASM1", "OA-ASM2", "OA-ASM3", "SMP143", "SMP142", 
              "OA-AL2i1", "OA-AL2i2", "OA-AL2i3", "OA-AL2i4", "OA-AL2b1", 
              "DNg34", "DNg104", "OA-VPM3", "OA-VPM4", "OA-VUMa1", 
              "OA-VUMa2", "OA-VUMa3", "OA-VUMa4", "OA-VUMa5", "OA-VUMa6", 
              "OA-VUMa8", "DNge138", "DNge149", "DNge152", "DNg66", 
              "DNge150", "DNge151", "DNd02", "DNd03"]
dataset = "fafb"       # Specify which dataset to read from.
alpha = 0.05            # Significance threshold.
use_adjusted = True     # Color by BH-adjusted p-values (q-values) rather than raw p-values.


def plot_pvalue_heatmap(cell_types, actual, nulls, modalities=MODALITIES, alpha=0.05, use_adjusted=True, ax=None):
    """
    Heatmap of one-tailed empirical p-values (proportion of null replicates >= the
    observed value) for each cell type x modality pair. Rows are cell types, columns
    are modalities, and squares are colored on a continuous 0-1 scale by p-value (or,
    if use_adjusted, by the Benjamini-Hochberg-adjusted q-value). BH correction is
    applied once across the full set of cell type x modality tests shown here. Squares
    that pass the significance threshold are marked with a "*".

    actual: {cell_type: {modality: value, ..., "total": value}}, e.g. loaded from
            {dataset}_actual_weighted_results.pkl.
    nulls: {cell_type: {modality: np.ndarray of null values, ..., "total": ndarray}},
           e.g. loaded from {dataset}_null_weighted_results.pkl.
    """
    n_rows, n_cols = len(cell_types), len(modalities)

    pvalues = np.array([
        [empirical_pvalue(actual[ct][mod], nulls[ct][mod]) for mod in modalities]
        for ct in cell_types
    ])
    qvalues = benjamini_hochberg(pvalues.ravel()).reshape(n_rows, n_cols)

    matrix = qvalues if use_adjusted else pvalues
    # Significance is always assessed on the FDR-adjusted q-value.
    significant = qvalues < alpha

    if ax is None:
        _, ax = plt.subplots(figsize=(1.2 * n_cols + 3, 0.5 * n_rows + 2))

    cmap = plt.get_cmap("Oranges_r")
    norm_obj = plt.Normalize(vmin=0.0, vmax=1.0)

    ax.imshow(matrix, cmap=cmap, norm=norm_obj, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            if significant[i, j]:
                rgba = cmap(norm_obj(matrix[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if lum > 0.5 else "white"
                ax.text(j, i, "*", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([mod.capitalize() for mod in modalities], rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(cell_types)

    label = "Adjusted p-value (q)" if use_adjusted else "p-value"
    title_suffix = "adjusted p-values (q)" if use_adjusted else "p-values"
    ax.set_title(f"Sensory influence significance ({title_suffix}), * = q < {alpha}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=label)

    return ax


def main(cell_types, dataset="fafb", alpha=0.05, use_adjusted=True):
    actual_path = NULL_CONNECTOMES_DIR / f"{dataset}_actual_weighted_results.pkl"
    null_path = NULL_CONNECTOMES_DIR / f"{dataset}_null_weighted_results.pkl"

    with open(str(actual_path), "rb") as file:
        actual = pickle.load(file)
    with open(str(null_path), "rb") as file:
        nulls = pickle.load(file)

    missing = [ct for ct in cell_types if ct not in actual]
    if missing:
        raise ValueError(f"{missing} not found in {actual_path.name}")

    fig, ax = plt.subplots(figsize=(1.2 * len(MODALITIES) + 3, 0.5 * len(cell_types) + 2))
    plot_pvalue_heatmap(cell_types, actual, nulls, alpha=alpha, use_adjusted=use_adjusted, ax=ax)
    fig.tight_layout()

    out_path = FIGS_DIR / f"{dataset}_pvalue_heatmap.png"
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else dataset
    main(CELL_TYPES, dataset, alpha, use_adjusted)

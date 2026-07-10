import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from connectomics.config.paths import NULL_CONNECTOMES_DIR, FIGS_DIR
from connectomics.config.consts import MODALITIES
from connectomics.modeling.significance import empirical_pvalue, benjamini_hochberg

cell_type = "SMP142" # Specify which cell type to plot.
dataset = "fafb"      # Specify which dataset to read from.
alpha = 0.05           # FDR significance threshold.

def plot_null_comparison(cell_type, actual, nulls, modalities=MODALITIES, alpha=0.05):
    """
    For one cell type, plot its null distribution of sensory input proportion against
    its actual (observed) proportion, one subplot per modality.

    Significance is assessed with a one-tailed empirical p-value per modality (the
    proportion of null replicates >= the observed value), then Benjamini-Hochberg FDR
    correction across the modalities shown in this figure (see
    connectomics.modeling.significance).

    actual: {cell_type: {modality: value, ..., "total": value}}, e.g. loaded from
            {dataset}_actual_weighted_results.pkl.
    nulls: {cell_type: {modality: np.ndarray of null values, ..., "total": ndarray}},
           e.g. loaded from {dataset}_null_weighted_results.pkl.
    """
    pvalues = [
        empirical_pvalue(actual[cell_type][mod], nulls[cell_type][mod])
        for mod in modalities
    ]
    qvalues = benjamini_hochberg(pvalues)

    ncols = 4
    nrows = -(-len(modalities) // ncols)  # ceil division
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    fig.suptitle(f"{cell_type} Sensory Inputs w/ Randomized Sensory Layers")
    flataxs = axs.flatten()

    for i, (mod, ax) in enumerate(zip(modalities, flataxs)):
        null_props = nulls[cell_type][mod]
        real_prop = actual[cell_type][mod]
        p, q = pvalues[i], qvalues[i]
        significant = q < alpha

        axvlabel = "Actual Proportion Sensory Input" if i == 0 else "__nolegend__"

        ax.hist(x=null_props, bins=20)
        ax.axvline(x=real_prop, color="r", linestyle="--", label=axvlabel)

        ax.set_xlabel(f"Null Proportion {mod.capitalize()} Input")
        if i == 0:
            ax.set_ylabel("Count")

        status = "significant" if significant else "n.s."
        ax.set_title(
            f"{status} (p={p:.3g}, q={q:.3g})",
            color="black",
            fontsize=10,
        )

    for ax in flataxs[len(modalities):]:
        ax.set_axis_off()

    fig.legend()
    fig.tight_layout()
    return fig


def main(cell_type, dataset="fafb", alpha=0.05):
    actual_path = NULL_CONNECTOMES_DIR / f"{dataset}_actual_weighted_results.pkl"
    null_path = NULL_CONNECTOMES_DIR / f"{dataset}_null_weighted_results.pkl"

    with open(str(actual_path), "rb") as file:
        actual = pickle.load(file)
    with open(str(null_path), "rb") as file:
        nulls = pickle.load(file)

    if cell_type not in actual:
        raise ValueError(f"'{cell_type}' not found in {actual_path.name}")

    fig = plot_null_comparison(cell_type, actual, nulls, alpha=alpha)

    out_path = FIGS_DIR / f"{dataset}_{cell_type}_sensory_null_comparison.png"
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    cell_type = sys.argv[1] if len(sys.argv) > 1 else cell_type
    dataset = sys.argv[2] if len(sys.argv) > 2 else dataset
    main(cell_type, dataset, alpha)

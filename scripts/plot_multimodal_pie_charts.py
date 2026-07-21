import sys
import pickle
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import NULL_CONNECTOMES_DIR, FIGS_DIR
from connectomics.config.consts import MODALITIES, MODALITY_COLORS
from connectomics.modeling.significance import empirical_pvalue, benjamini_hochberg

dataset = "fafb"  # Specify which dataset to read from.
alpha = 0.05       # FDR significance threshold.

# Slices smaller than this fraction of their pie don't get an in-slice count label
# (it wouldn't fit); the count is still available in the legend.
MIN_LABEL_FRACTION = 0.015


def significant_modalities_per_type(actual, nulls, dataset, modalities=MODALITIES, alpha=0.05):
    """
    For each non-sensory cell type, determines which modalities show a statistically
    significant weighted sensory-input proportion.

    Significance is assessed per cell type with a one-tailed empirical p-value per
    modality (proportion of null replicates >= the observed value), then
    Benjamini-Hochberg FDR-corrected across the `modalities` tested for that cell type
    (n=len(modalities)), matching plot_sensory_null_comparison.py and
    plot_significant_types_per_class.py.

    Cell types belonging to a sensory super_class are excluded, since this is about
    downstream sensory input, not the sensory neurons that are the source of it.

    actual: {cell_type: {modality: value, ..., "total": value}}, e.g. loaded from
            {dataset}_actual_weighted_results.pkl.
    nulls: {cell_type: {modality: np.ndarray of null values, ..., "total": ndarray}},
           e.g. loaded from {dataset}_null_weighted_results.pkl.

    Returns: {cell_type: frozenset of significant modalities}, omitting cell types with
    zero significant modalities.
    """
    celltypes_df = DATASETS[dataset].celltypes_df
    classifications_df = DATASETS[dataset].classifications_df

    sensory_classes = set(
        classifications_df.loc[classifications_df["super_class"] == "sensory", "class"].unique()
    )

    root_by_type = celltypes_df.drop_duplicates("primary_type").set_index("primary_type")["root_id"]
    class_by_root = classifications_df.set_index("root_id")["class"]

    sig_by_type = {}
    for cell_type in actual:
        if cell_type not in root_by_type.index:
            print(f"Warning: '{cell_type}' does not exist in dataset.")
            continue

        root_id = root_by_type[cell_type]
        class_ = class_by_root.get(root_id)
        if class_ in sensory_classes:
            continue

        pvalues = [empirical_pvalue(actual[cell_type][mod], nulls[cell_type][mod]) for mod in modalities]
        qvalues = benjamini_hochberg(pvalues)

        combo = frozenset(mod for mod, q in zip(modalities, qvalues) if q < alpha)
        if combo:
            sig_by_type[cell_type] = combo

    return sig_by_type


def combo_counts_by_n(sig_by_type, modalities=MODALITIES):
    """
    Groups cell types by how many modalities they are significant for (n), then by the
    specific combination of modalities within each n.

    Returns: {n: Counter({combo: count})}, only for n present in the data. combo is a
    tuple of modality names ordered to match `modalities`.
    """
    grouped = defaultdict(Counter)
    for combo in sig_by_type.values():
        ordered = tuple(mod for mod in modalities if mod in combo)
        grouped[len(ordered)][ordered] += 1
    return grouped


def combo_color(combo, modality_colors=MODALITY_COLORS):
    """Averages the constituent modalities' colors, so related combinations read as
    related colors (e.g. any combo including visual leans toward visual's hue)."""
    rgbs = [mcolors.to_rgb(modality_colors[mod]) for mod in combo]
    return tuple(np.mean(rgbs, axis=0))


def combo_label(combo):
    return "+".join(mod.capitalize() for mod in combo)


def plot_multimodal_pie(n, counts, dataset="fafb", alpha=0.05):
    """
    One pie chart for cell types significant for exactly `n` modalities. Each wedge is
    one combination of modalities; wedge size = number of cell types with exactly that
    combination significant.

    In-slice labels are the cell-type count (omitted for slivers too small to render
    legibly); the legend to the right gives the modality combination + count for every
    wedge, since that's more legible than radial labels once there are more than a
    handful of combinations (e.g. up to C(7,2)=21 for n=2).

    counts: Counter({combo: count}), combo a tuple of modality names.
    """
    combos = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    sizes = [count for _, count in combos]
    total = sum(sizes)
    colors = [combo_color(combo) for combo, _ in combos]
    legend_labels = [f"{combo_label(combo)} (n={count})" for combo, count in combos]

    fig, ax = plt.subplots(figsize=(9, 6))

    def autopct(pct):
        return str(int(round(pct * total / 100))) if pct / 100 >= MIN_LABEL_FRACTION else ""

    wedges, _, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct=autopct,
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 9},
    )

    # Recolor each count label for contrast against its wedge.
    for wedge, autotext in zip(wedges, autotexts):
        r, g, b = wedge.get_facecolor()[:3]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        autotext.set_color("black" if lum > 0.5 else "white")

    modality_word = "modality" if n == 1 else "modalities"
    ax.set_title(
        f"Types significant for {n} {modality_word} (n={total})\n"
        f"{dataset}, α={alpha}"
    )
    ax.legend(
        wedges,
        legend_labels,
        title="Modality combination",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title_fontsize=9,
    )
    fig.tight_layout()
    return fig


def main(dataset="fafb", alpha=0.05):
    actual_path = NULL_CONNECTOMES_DIR / f"{dataset}_actual_weighted_results.pkl"
    null_path = NULL_CONNECTOMES_DIR / f"{dataset}_null_weighted_results.pkl"

    with open(str(actual_path), "rb") as file:
        actual = pickle.load(file)
    with open(str(null_path), "rb") as file:
        nulls = pickle.load(file)

    sig_by_type = significant_modalities_per_type(actual, nulls, dataset, alpha=alpha)
    grouped = combo_counts_by_n(sig_by_type)

    for n in sorted(grouped.keys()):
        fig = plot_multimodal_pie(n, grouped[n], dataset=dataset, alpha=alpha)
        out_path = FIGS_DIR / f"{dataset}_multimodal_pie_n{n}.png"
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else dataset
    main(dataset, alpha)

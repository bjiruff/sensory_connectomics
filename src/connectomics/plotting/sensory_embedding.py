import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from connectomics.config.consts import MODALITIES, MODALITY_COLORS

MARKER_SIZE = 8
ALPHA = 0.75


def super_class_colors(super_class):
    """
    Maps each super_class to a fixed color from tab20 (same convention as
    plot_type_umap.py's _get_type_colors), so a given super_class reads as the
    same color across figures.

    Returns (colors list aligned to `super_class`, {super_class: color} dict
    for the legend).
    """
    unique_classes = sorted(sc for sc in super_class.unique() if sc != "Unknown")
    cmap = plt.colormaps["tab20"].resampled(max(len(unique_classes), 1))
    class_colormap = {sc: cmap(i) for i, sc in enumerate(unique_classes)}
    class_colormap["Unknown"] = (0.6, 0.6, 0.6, 1.0)

    colors = [class_colormap[sc] for sc in super_class]
    return colors, class_colormap


def _modality_cmap(modality):
    """Light-to-dark sequential colormap in the modality's own brand color
    (MODALITY_COLORS), so each panel ties visually to that modality's color
    elsewhere in the project (e.g. plot_multimodal_pie_charts.py)."""
    return LinearSegmentedColormap.from_list(f"{modality}_seq", ["#f2f2f2", MODALITY_COLORS[modality]])


def _add_biplot_vectors(ax, embedding, loadings):
    """
    Overlays PCA loading vectors on top of the score scatter already drawn on
    `ax`, turning it into a biplot. One arrow per modality, colored to match
    MODALITY_COLORS for a consistent visual identity with the other panels.

    loadings: DataFrame (index = modality names, columns = the 2 embedding
    dims), e.g. `pca.components_.T`. Each PC's loadings are a unit vector
    across modalities, so they are rescaled here to visually span roughly the
    same extent as the score scatter (standard biplot convention) — only
    their relative lengths and directions are meaningful, not their absolute
    scale against the score axes.
    """
    scores_range = np.abs(embedding).max()
    loadings_range = np.abs(loadings.to_numpy()).max()
    scale = 0.8 * scores_range / loadings_range if loadings_range > 0 else 1.0
    vectors = loadings.to_numpy() * scale

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    margin_x, margin_y = 0.05 * (xlim[1] - xlim[0]), 0.05 * (ylim[1] - ylim[0])
    tangential_step = 0.09 * min(xlim[1] - xlim[0], ylim[1] - ylim[0])

    # Vectors pointing in a similar direction (e.g. thermo- and hygrosensory
    # here) would otherwise stack their labels directly on top of each other.
    # Group consecutive-by-angle vectors into clusters, then within each
    # cluster fan the labels out sideways (perpendicular to the cluster's mean
    # direction) instead of just further along the same ray, which barely
    # separates near-parallel vectors.
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    order = np.argsort(angles)
    angle_close = np.radians(25)

    clusters = [[order[0]]]
    for idx in order[1:]:
        if (angles[idx] - angles[clusters[-1][-1]]) < angle_close:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    label_pos = {}
    for cluster in clusters:
        mean_angle = np.mean(angles[cluster])
        perp = np.array([-np.sin(mean_angle), np.cos(mean_angle)])
        n = len(cluster)
        offsets = (np.arange(n) - (n - 1) / 2) * tangential_step
        for member, offset in zip(cluster, offsets):
            label_pos[member] = vectors[member] * 1.2 + perp * offset

    for i, modality in enumerate(loadings.index):
        dx, dy = vectors[i]
        color = MODALITY_COLORS[modality]
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=15),
            zorder=5,
        )
        lx, ly = label_pos[i]
        # Belt-and-suspenders: whatever the offset, keep the label within the
        # axes (with a small margin) so a long vector near the edge can't
        # push its label out into the figure margin or an adjacent panel.
        lx = np.clip(lx, xlim[0] + margin_x, xlim[1] - margin_x)
        ly = np.clip(ly, ylim[0] + margin_y, ylim[1] - margin_y)
        ax.text(
            lx, ly, modality.capitalize(), color=color,
            fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        )


def plot_embedding_grid(embedding, raw_proportions, super_class, method_name, dataset="fafb", axis_labels=("Dim 1", "Dim 2"), loadings=None):
    """
    Lays out 8 scatter panels of the same 2D embedding on one figure: one
    colored by super_class annotation, and one per modality (in MODALITIES
    order) colored by that modality's raw weighted-sensory-input proportion —
    the untransformed values, not the log-ratio features the embedding itself
    was computed from — so clustering can be checked against the original
    quantity of interest.

    embedding: (n_cells, 2) array, row-aligned with `raw_proportions` and
               `super_class`.
    raw_proportions: DataFrame (n_cells, len(MODALITIES)).
    super_class: Series of super_class annotations.
    method_name: e.g. "PCA" or "UMAP", used in the figure title.
    axis_labels: (x_label, y_label) applied to every panel.
    loadings: optional DataFrame (index = modality names, columns = the 2
        embedding dims) of per-modality loadings, e.g. `pca.components_.T`.
        When given, the super_class panel becomes a biplot with one arrow per
        modality showing its contribution to the two axes. Only meaningful
        for a linear embedding (PCA); leave as None for UMAP.

    Returns the created Figure.
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.ravel()

    colors, class_colormap = super_class_colors(super_class)
    ax = axes[0]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=MARKER_SIZE, alpha=ALPHA, linewidths=0)
    ax.set_title("Super class (biplot)" if loadings is not None else "Super class")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    if loadings is not None:
        _add_biplot_vectors(ax, embedding, loadings)

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", label=sc, markerfacecolor=color, markersize=7)
        for sc, color in class_colormap.items()
    ]
    fig.legend(
        handles=legend_elements, title="Super class", loc="lower center",
        ncol=len(legend_elements), bbox_to_anchor=(0.5, 0.0), fontsize=8, title_fontsize=9,
    )

    for i, modality in enumerate(MODALITIES, start=1):
        ax = axes[i]
        values = raw_proportions[modality].to_numpy()
        # Per-modality 99th-percentile ceiling: modalities differ widely in
        # typical magnitude, so a shared vmax would wash out the quieter ones.
        vmax = max(np.quantile(values, 0.99), 1e-6)

        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=values, cmap=_modality_cmap(modality),
            vmin=0.0, vmax=vmax, s=MARKER_SIZE, alpha=ALPHA, linewidths=0,
        )
        ax.set_title(modality.capitalize())
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Raw proportion")

    fig.suptitle(f"{method_name} of cell-type sensory-input composition ({dataset})", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig

import sys
import matplotlib.pyplot as plt
from umap import UMAP
from connectomics.processing.sensory_embedding import load_modality_proportions, clr_transform, get_super_class
from connectomics.plotting.sensory_embedding import plot_embedding_grid
from connectomics.config.paths import FIGS_DIR


def compute_umap_embedding(clr_df, random_state=42):
    """
    UMAP on the CLR-transformed modality composition, using the default
    Euclidean metric. Euclidean distance between CLR-transformed rows equals
    Aitchison distance, the standard distance for compositional data, so no
    custom metric is needed here (unlike plot_type_umap.py, which uses cosine
    distance over raw connectivity count vectors).
    """
    reducer = UMAP(n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(clr_df.to_numpy())
    return embedding


def main(dataset="fafb"):
    raw_df = load_modality_proportions(dataset)
    clr_df = clr_transform(raw_df)

    embedding = compute_umap_embedding(clr_df)
    super_class = get_super_class(dataset, raw_df.index)

    fig = plot_embedding_grid(embedding, raw_df, super_class, method_name="UMAP", dataset=dataset, axis_labels=("UMAP 1", "UMAP 2"))

    out_path = FIGS_DIR / f"{dataset}_sensory_umap.png"
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fafb"
    main(dataset)

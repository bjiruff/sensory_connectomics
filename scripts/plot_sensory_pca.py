import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from connectomics.processing.sensory_embedding import load_modality_proportions, clr_transform, get_super_class
from connectomics.plotting.sensory_embedding import plot_embedding_grid
from connectomics.config.paths import FIGS_DIR


def compute_pca_embedding(clr_df, random_state=42):
    """PCA on the CLR-transformed modality composition. Unlike UMAP, this is
    deterministic given the input; random_state only affects the internal SVD
    solver's tie-breaking."""
    reducer = PCA(n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(clr_df.to_numpy())
    return embedding, reducer


def main(dataset="fafb"):
    raw_df = load_modality_proportions(dataset)
    clr_df = clr_transform(raw_df)

    embedding, reducer = compute_pca_embedding(clr_df)
    super_class = get_super_class(dataset, raw_df.index)

    explained = reducer.explained_variance_ratio_
    axis_labels = (f"PC1 ({explained[0]:.1%} var.)", f"PC2 ({explained[1]:.1%} var.)")
    loadings = pd.DataFrame(reducer.components_.T, index=clr_df.columns, columns=["PC1", "PC2"])
    fig = plot_embedding_grid(
        embedding, raw_df, super_class, method_name="PCA", dataset=dataset,
        axis_labels=axis_labels, loadings=loadings,
    )

    out_path = FIGS_DIR / f"{dataset}_sensory_pca.png"
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)
    print(f"PC1 explains {explained[0]:.1%} of variance, PC2 explains {explained[1]:.1%}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fafb"
    main(dataset)

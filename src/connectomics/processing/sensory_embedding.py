import pickle
import numpy as np
import pandas as pd
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import NULL_CONNECTOMES_DIR
from connectomics.config.consts import MODALITIES


def load_modality_proportions(dataset="fafb", modalities=MODALITIES):
    """
    Loads {dataset}_actual_weighted_results.pkl and returns the raw per-modality
    weighted sensory-input proportions as a DataFrame (rows = cell type, columns
    = `modalities`). The "total" field is intentionally excluded: it is the
    overall magnitude of traceable sensory drive, not part of the modality
    composition.

    Cell types with no traceable input from any modality (all-zero rows, ~1%
    of types) are dropped: a composition is undefined for the all-zero vector,
    and keeping them would collapse many unrelated types onto one meaningless
    point once log-ratio transformed.
    """
    path = NULL_CONNECTOMES_DIR / f"{dataset}_actual_weighted_results.pkl"
    with open(str(path), "rb") as file:
        actual = pickle.load(file)

    df = pd.DataFrame.from_dict(actual, orient="index")[modalities].astype(np.float64)
    return df.loc[(df != 0).any(axis=1)]


def clr_transform(df):
    """
    Centered log-ratio (CLR) transform: unconstrains compositional data (parts
    of a whole, here modality proportions) onto real-valued coordinates so that
    Euclidean-geometry methods like PCA/UMAP become meaningful.

    CLR is scale-invariant, so rows need not be pre-normalized to sum to 1:
    log(x_i / geometric_mean(x)) is unchanged by row-wise rescaling. A useful
    side effect is that Euclidean distance between CLR-transformed rows equals
    Aitchison distance, the standard distance metric for compositional data.

    Exact zeros are undefined under the log and are replaced with a small
    pseudocount before transforming: 65% of the smallest nonzero value observed
    anywhere in the data, a simple multiplicative zero-replacement convention
    (Martin-Fernandez et al.) that only nudges the affected parts.

    Returns a DataFrame of the same shape/index/columns as `df`.
    """
    values = df.to_numpy(dtype=np.float64)
    nonzero = values[values > 0]
    pseudocount = nonzero.min() * 0.65 if nonzero.size else 1e-6
    values = np.where(values <= 0, pseudocount, values)

    log_values = np.log(values)
    clr = log_values - log_values.mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=df.index, columns=df.columns)


def get_super_class(dataset, cell_types):
    """
    Looks up each cell type's super_class annotation via a representative
    root_id (the first cell of that primary_type in celltypes_df), matching
    the convention used elsewhere for actual_weighted_results-keyed scripts
    (e.g. plot_significant_types_per_class.py).

    Returns a Series aligned to `cell_types`, "Unknown" where unavailable.
    """
    celltypes_df = DATASETS[dataset].celltypes_df
    classifications_df = DATASETS[dataset].classifications_df

    root_by_type = celltypes_df.drop_duplicates("primary_type").set_index("primary_type")["root_id"]
    super_class_by_root = classifications_df.set_index("root_id")["super_class"]

    root_ids = pd.Index(cell_types).map(root_by_type)
    super_class = root_ids.map(super_class_by_root)
    return pd.Series(super_class, index=cell_types).fillna("Unknown")

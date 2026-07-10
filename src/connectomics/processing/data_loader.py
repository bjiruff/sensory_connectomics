from pathlib import Path
import pandas as pd
from connectomics.config.consts import VALID_DATASETS
from connectomics.config.paths import CONNECTOMES_DIR

class Dataset:
    def __init__(
            self,
            cell_connections_df,
            type_connections_df,
            celltypes_df,
            classifications_df,
            neurotransmitters_df
            ):
        self.cell_connections_df = cell_connections_df
        self.type_connections_df = type_connections_df
        self.celltypes_df = celltypes_df
        self.classifications_df = classifications_df
        self.neurotransmitters_df = neurotransmitters_df

def connectome_exists(c):
    required_files = [
        f"{c}_cell_connections.csv",
        f"{c}_type_connections.csv",
        f"{c}_celltypes.csv",
        f"{c}_classifications.csv",
        f"{c}_neurotransmitters.csv"
    ]
    missing = [f for f in required_files if not (CONNECTOMES_DIR / f).is_file()]

    if missing:
        missing_list = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Connectome '{c}' is incomplete. Missing files:\n  {missing_list}"
        )

DATASETS = {}
for c in VALID_DATASETS:
    connectome_exists(c)
    DATASETS[c] = Dataset(
        pd.read_csv(CONNECTOMES_DIR / f"{c}_cell_connections.csv"),
        pd.read_csv(CONNECTOMES_DIR / f"{c}_type_connections.csv"),
        pd.read_csv(CONNECTOMES_DIR / f"{c}_celltypes.csv"),
        pd.read_csv(CONNECTOMES_DIR / f"{c}_classifications.csv"),
        pd.read_csv(CONNECTOMES_DIR / f"{c}_neurotransmitters.csv", dtype={"nt_type": "object"}),
    )
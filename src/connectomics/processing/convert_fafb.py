import shutil
import pandas as pd
from connectomics.config.paths import RAW_DIR, CONNECTOMES_DIR

# Renaming FAFB sheets and doing minor processing on connections data.

# Download consolidated_cell_types.csv, classification.csv, neurons.csv, connections_princeton.csv from https://codex.flywire.ai/api/download?dataset=fafb
def main():
    celltypes_df = pd.read_csv(RAW_DIR / "consolidated_cell_types.csv")
    celltypes_df.to_csv(CONNECTOMES_DIR / "fafb_celltypes.csv", index=False)
    print("fafb_celltypes.csv created")

    shutil.copy(RAW_DIR / "classification.csv", CONNECTOMES_DIR / "fafb_classifications.csv")
    print("fafb_classifications.csv created")

    shutil.copy(RAW_DIR / "neurons.csv", CONNECTOMES_DIR / "fafb_neurotransmitters.csv")
    print("fafb_neurotransmitters.csv created")

    raw_cell_connections_df = pd.read_csv(RAW_DIR / "connections_princeton.csv")
    cell_connections_df = raw_cell_connections_df.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    cell_connections_df.to_csv(CONNECTOMES_DIR / "fafb_cell_connections.csv", index=False)
    print("fafb_cell_connections.csv created")

    id_to_type = celltypes_df.set_index("root_id")["primary_type"]
    type_connections_df = cell_connections_df.copy().rename(columns={"pre_root_id": "pre_type", "post_root_id": "post_type"})
    type_connections_df["pre_type"] = type_connections_df["pre_type"].map(id_to_type)
    type_connections_df["post_type"] = type_connections_df["post_type"].map(id_to_type)
    type_connections_df = type_connections_df.groupby(["pre_type", "post_type"], as_index=False)["syn_count"].sum()
    type_connections_df.to_csv(CONNECTOMES_DIR / "fafb_type_connections.csv", index=False)
    print("fafb_type_connections.csv created")

if __name__ == "__main__":
    main()
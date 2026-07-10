import pandas as pd
from connectomics.config.paths import RAW_DIR, CONNECTOMES_DIR

# Converting MCNS to FAFB-like sheets.

def main():
    # Downloaded from https://male-cns.janelia.org/download/#__tabbed_3_3
    raw_annotations_df = pd.read_feather(RAW_DIR / "body-annotations-male-cns-v0.9-minconf-0.5.feather")
    raw_cell_connections_df = pd.read_feather(RAW_DIR / "connectome-weights-male-cns-v0.9-minconf-0.5.feather")
    raw_neurotransmitters_df = pd.read_feather(RAW_DIR / "body-neurotransmitters-male-cns-v0.9.feather")

    # Make cell types csv.
    celltypes_df = raw_annotations_df[["bodyId", "type"]]
    celltypes_df = celltypes_df.rename(columns={"bodyId": "root_id", "type": "primary_type"})
    celltypes_df = celltypes_df[celltypes_df["primary_type"].notna()]
    # Some cell types have commas to denote that the neuron is assigned to more than one cell. We only want the primary type.
    celltypes_df["primary_type"] = celltypes_df["primary_type"].apply(lambda x: x.split(",")[0])
    celltypes_df.to_csv(CONNECTOMES_DIR / "mcns_celltypes.csv", index=False)
    print("mcns_celltypes.csv created")

    # Make classifications csv.
    classifications_df = raw_annotations_df[["bodyId", "superclass", "class", "subclass", "itoleeHl", "somaSide"]]
    classifications_df = classifications_df.rename(columns={
        "bodyId": "root_id",
        "superclass": "super_class",
        "subclass": "sub_class",
        "itoleeHl": "hemilineage",
        "somaSide": "side"
        })
    classifications_df["side"] = (
        classifications_df["side"]
        .replace({
            "L": "left",
            "R": "right",
            "M": "center"
        })
    )
    classifications_df.to_csv(CONNECTOMES_DIR / "mcns_classifications.csv", index=False)
    print("mcns_classifications.csv created")

    # Make neurotransmitters csv.
    neurotransmitters_df = raw_neurotransmitters_df[["body", "predicted_nt", "predicted_nt_confidence"]]
    neurotransmitters_df = neurotransmitters_df.rename(columns={
        "body": "root_id",
        "predicted_nt": "nt_type",
        "predicted_nt_confidence": "nt_type_score"
    })
    neurotransmitters_df["nt_type"] = (
        neurotransmitters_df["nt_type"]
        .replace({
            "acetylcholine": "ACH",
            "dopamine": "DA",
            "gaba": "GABA",
            "glutamate": "GLUT",
            "histamine": "HIST",
            "octopamine": "OCT",
            "serotonin": "SER",
            "unclear": pd.NA
        })
    )
    neurotransmitters_df = neurotransmitters_df.astype({
        "nt_type": "object"
    })
    neurotransmitters_df.to_csv(CONNECTOMES_DIR / "mcns_neurotransmitters.csv", index=False)
    print("mcns_neurotransmitters.csv created")

    # Make connections csv.
    # Make separate csvs for all traced connections and all traced connections with synapse count > 5.
    raw_cell_connections_df = raw_cell_connections_df[raw_cell_connections_df["weight"] >= 5]

    traced_bodies = set(
        raw_annotations_df[raw_annotations_df["status"] == "Traced"]["bodyId"]
    )

    cell_connections_df = raw_cell_connections_df[
        raw_cell_connections_df["body_pre"].isin(traced_bodies) &
        raw_cell_connections_df["body_post"].isin(traced_bodies)
    ][["body_pre", "body_post", "weight"]].rename(columns={
        "body_pre": "pre_root_id",
        "body_post": "post_root_id",
        "weight": "syn_count"
    })

    cell_connections_df.to_csv(CONNECTOMES_DIR / "mcns_cell_connections.csv", index=False)
    print("mcns_cell_connections.csv created")

    id_to_type = celltypes_df.set_index("root_id")["primary_type"]
    type_connections_df = cell_connections_df.copy().rename(columns={"pre_root_id": "pre_type", "post_root_id": "post_type"})
    type_connections_df["pre_type"] = type_connections_df["pre_type"].map(id_to_type)
    type_connections_df["post_type"] = type_connections_df["post_type"].map(id_to_type)
    type_connections_df = type_connections_df.groupby(["pre_type", "post_type"], as_index=False)["syn_count"].sum()
    type_connections_df.to_csv(CONNECTOMES_DIR / "mcns_type_connections.csv", index=False)
    print("mcns_type_connections.csv created")


if __name__ == "__main__":
    main()
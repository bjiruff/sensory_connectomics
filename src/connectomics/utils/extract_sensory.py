import pandas as pd
from connectomics.processing.data_loader import DATASETS
from connectomics.config.consts import VALID_DATASETS, MODALITIES


def _extract_typed(ids, celltypes_df):
    types = celltypes_df.loc[celltypes_df["root_id"].isin(ids), "primary_type"].unique()
    typed_ids = celltypes_df.loc[celltypes_df["primary_type"].isin(types), "root_id"]
    return typed_ids, types

def _get_sensory(dataset):

    if dataset not in VALID_DATASETS:
        raise ValueError(f"Specified dataset is invalid. Must be one of the following: {VALID_DATASETS}")

    classifications_df = DATASETS[dataset].classifications_df
    celltypes_df = DATASETS[dataset].celltypes_df

    gustatory_ids = classifications_df.loc[classifications_df["class"] == "gustatory", "root_id"]
    olfactory_ids = classifications_df.loc[classifications_df["class"] == "olfactory", "root_id"]
    thermo_ids = classifications_df.loc[classifications_df["class"] == "thermosensory", "root_id"]
    hygro_ids = classifications_df.loc[classifications_df["class"] == "hygrosensory", "root_id"]
    visual_ids = classifications_df.loc[classifications_df["class"] == "visual", "root_id"]
    mechano_ids = classifications_df.loc[
        (classifications_df["class"] == "mechanosensory") & ~(classifications_df["sub_class"] == "auditory"),
        "root_id"
    ]
    auditory_ids = classifications_df.loc[classifications_df["sub_class"] == "auditory", "root_id"]

    sensory_ids = {
        "gustatory": gustatory_ids,
        "olfactory": olfactory_ids,
        "thermosensory": thermo_ids,
        "hygrosensory": hygro_ids,
        "visual": visual_ids,
        "mechanosensory": mechano_ids,
        "auditory": auditory_ids
    }
    sensory_types = {}

    for modality in MODALITIES:
        ids = sensory_ids[modality]
        typed_ids, types = _extract_typed(ids, celltypes_df)
        sensory_ids[modality] = typed_ids
        sensory_types[modality] = types

    return sensory_ids, sensory_types

def _get_nonsensory_type_edges(dataset):
    """
    Constant non-sensory -> non-sensory type-level edges for `dataset`, i.e. the part of the
    connectome that is identical across every null connectome built by
    scripts/build_null_connectomes.py (only sensory -> second order edges are shuffled there).

    Returns a list of (pre_type, post_type, {"weight": syn_count}) tuples.
    """
    type_connections_df = DATASETS[dataset].type_connections_df
    _, sensory_types = _get_sensory(dataset)

    all_sensory_types = [t for types in sensory_types.values() for t in types]

    nonsensorytype_edge_df = type_connections_df.loc[
        (type_connections_df["pre_type"] != type_connections_df["post_type"]) &
        ~(type_connections_df["pre_type"].isin(all_sensory_types))
    ]
    return [
        (row.pre_type, row.post_type, {"weight": row.syn_count})
        for row in nonsensorytype_edge_df.itertuples()
    ]

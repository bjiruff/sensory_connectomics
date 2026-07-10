import networkx as nx
import numpy as np
import pickle
import os
from multiprocessing import Pool
import connectomics.utils.extract_sensory as extract_sensory
from connectomics.processing.data_loader import DATASETS
from connectomics.config.consts import VALID_DATASETS
from connectomics.config.paths import NULL_CONNECTOMES_DIR

dataset = "fafb" # Specify which dataset you want to build null connectomes out of.
num_nulls = 100
if dataset not in VALID_DATASETS:
    raise ValueError(f"Specified dataset is invalid. Must be one of the following: {VALID_DATASETS}")

cell_connections_df = DATASETS[dataset].cell_connections_df
celltypes_df = DATASETS[dataset].celltypes_df
id_to_type = celltypes_df[["root_id", "primary_type"]].set_index("root_id")["primary_type"]

# Deprecated, from when we were running propagation on full null connectomes.
# import diffusion_models
# modalities = ["gustatory", "olfactory", "thermosensory", "hygrosensory", "visual", "mechanosensory", "auditory"]
# propagation_method = diffusion_models.compute_sensory_proportions
# args = {
#     "epsilon": 0.001
# }

_worker_base_edge_df = None
_worker_nonsensory_idx = None
_worker_nonsensory_edges = None

def _worker_init(base_edge_df, nonsensory_idx, nonsensory_edges):
    global _worker_base_edge_df, _worker_nonsensory_idx, _worker_nonsensory_edges
    _worker_base_edge_df = base_edge_df
    _worker_nonsensory_idx = nonsensory_idx
    _worker_nonsensory_edges = nonsensory_edges
    os.environ["OMP_NUM_THREADS"] = "1"

def _run_one(seed):
    # Shuffle sensory to non-sensory and merge it with the non-sensory to non-sensory list to create a null connectome with randomized sensory inputs.

    # Shuffle sensory cell ids to non-sensory cell types.
    np.random.seed(seed)
    edge_df = _worker_base_edge_df.copy()
    post_types = edge_df.loc[_worker_nonsensory_idx, "post_type"].to_numpy(dtype=str, copy=True)
    np.random.shuffle(post_types)
    edge_df.loc[_worker_nonsensory_idx, "post_type"] = post_types

    # Consolidate sensory cell ids into their cell types.
    # Result is an edge list that contains randomized connections from sensory types to second order types.
    sensorytype_edge_df = edge_df.rename(columns={"pre_root_id": "pre_type"})
    sensorytype_edge_df["pre_type"] = sensorytype_edge_df["pre_type"].map(id_to_type)
    sensorytype_edge_df = sensorytype_edge_df.groupby(["pre_type", "post_type"], as_index=False)["syn_count"].sum()
    sensorytype_edge_df = sensorytype_edge_df.loc[
        sensorytype_edge_df["pre_type"] != sensorytype_edge_df["post_type"]
    ]

    edges = [
        (row.pre_type, row.post_type, {"weight": row.syn_count})
        for row in sensorytype_edge_df.itertuples()
    ]

    # Deprecated, adding non-sensory to non-sensory edges is redundant because they are the same with each null connectome.
    # Add in non-sensory to non-sensory edges and create the full null connectome.
    # edges += _worker_nonsensory_edges
    # null_G = nx.DiGraph()
    # null_G.add_edges_from(edges)
    # return propagation_method(null_G, data.sensory_nodes_by_modality, **args)

    return edges


if __name__ == "__main__":
    # Overall idea is to get separate edge lists for sensory to non-sensory and non-sensory to non-sensory, so we can shuffle the former without interfering with the latter.
    sensory_ids, sensory_types = extract_sensory._get_sensory(dataset)

    all_sensory_types = []
    for nodes in sensory_types.values():
        all_sensory_types.extend(nodes)

    all_sensory_ids = []
    for ids in sensory_ids.values():
        all_sensory_ids.extend(ids)

    # Precompute the base edge dataframe once. Constant every iteration.
    # base_edge_df contains connections from sensory cell ids to downstream cell types.
    base_edge_df = cell_connections_df.loc[
        cell_connections_df["pre_root_id"].isin(all_sensory_ids)
    ].copy()
    base_edge_df = base_edge_df.loc[base_edge_df["pre_root_id"].isin(id_to_type.index)] # Get only pre_root_id's with cell type annotations.
    base_edge_df["post_root_id"] = base_edge_df["post_root_id"].map(id_to_type)
    base_edge_df.rename(columns={"post_root_id": "post_type"}, inplace=True)
    base_edge_df = base_edge_df.groupby(["pre_root_id", "post_type"], as_index=False)["syn_count"].sum()

    # Precompute which rows target non-sensory nodes. Constant every iteration.
    nonsensory_mask = ~base_edge_df["post_type"].isin(all_sensory_types)
    nonsensory_idx = base_edge_df.index[nonsensory_mask].to_numpy()

    # Precompute edge list for cell types where pre_type is not sensory. Constant every iteration,
    # and shared with scripts/run_null_diffuse.py and scripts/run_null_bernoulli.py.
    nonsensory_edges = extract_sensory._get_nonsensory_type_edges(dataset)

    nulls = None
    completed = 0

    with Pool(
        initializer=_worker_init,
        initargs=(base_edge_df, nonsensory_idx, nonsensory_edges)
    ) as pool:
        for result in pool.imap_unordered(_run_one, range(num_nulls)):
            # Deprecated, from when we were running propagation on full null connectomes.
            # if nulls is None:
            #     nulls = {node: {m: [] for m in modalities} for node in result}
            # for node, dic in result.items():
            #     for mod in modalities:
            #         nulls[node][mod].append(dic[mod])

            if nulls is None:
                nulls = []
            nulls.append(result)

            completed += 1
            if completed % 50 == 0:
                print(f"Finished {completed} iterations")

    # Deprecated, from when we were running propagation on full null connectomes.
    # for node, dic in nulls.items():
    #     for mod, vals in dic.items():
    #         dic[mod] = np.array(vals, dtype=np.float32)

    with open(str(NULL_CONNECTOMES_DIR / f"{dataset}_null_sensory_edges.pkl"), "wb") as file:
        pickle.dump(nulls, file)

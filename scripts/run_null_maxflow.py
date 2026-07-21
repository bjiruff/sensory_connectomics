import sys
import pickle
import networkx as nx
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import NULL_CONNECTOMES_DIR
from connectomics.utils.extract_sensory import _get_sensory, _get_nonsensory_type_edges
from connectomics.modeling.flow_models import maxflow_sensory_analysis
from connectomics.modeling.null_comparison import run_model_on_nulls

cell_type = "SMP143" # Specify which cell type (maxflow target) to analyze.
dataset = "fafb"      # Specify which dataset to run the analysis on.
max_hops = 4           # Maximum path length (in hops) to include in the flow.

def main(cell_type, dataset="fafb", max_hops=3):
    _, sensory_types = _get_sensory(dataset)

    # Build the real connectome graph at the cell-type level, matching the granularity of the
    # null connectomes built by scripts/build_null_connectomes.py.
    type_connections_df = DATASETS[dataset].type_connections_df
    real_edges = [
        (row.pre_type, row.post_type, {"weight": row.syn_count})
        for row in type_connections_df.itertuples()
        if row.pre_type != row.post_type
    ]
    G = nx.DiGraph()
    G.add_edges_from(real_edges)

    if cell_type not in G:
        raise ValueError(f"'{cell_type}' not found in the {dataset} connectome.")

    print(f"Running maxflow_sensory_analysis (<= {max_hops} hops) on the real connectome...")
    actual = maxflow_sensory_analysis(G, sensory_types, target=cell_type, max_hops=max_hops)

    with open(str(NULL_CONNECTOMES_DIR / f"{dataset}_null_sensory_edges_1000.pkl"), "rb") as file:
        null_edge_sets = pickle.load(file)

    nonsensory_edges = _get_nonsensory_type_edges(dataset)

    print(f"Running maxflow_sensory_analysis over {len(null_edge_sets)} null connectomes...")
    nulls = run_model_on_nulls(
        maxflow_sensory_analysis,
        null_edge_sets,
        nonsensory_edges,
        sensory_types,
        model_kwargs={"target": cell_type, "max_hops": max_hops},
    )

    actual_path = NULL_CONNECTOMES_DIR / f"{dataset}_{cell_type}_actual_maxflow_results.pkl"
    null_path = NULL_CONNECTOMES_DIR / f"{dataset}_{cell_type}_null_maxflow_results.pkl"
    with open(str(actual_path), "wb") as file:
        pickle.dump(actual, file)
    with open(str(null_path), "wb") as file:
        pickle.dump(nulls, file)
    print(f"Saved results to {NULL_CONNECTOMES_DIR}")

if __name__ == "__main__":
    cell_type = sys.argv[1] if len(sys.argv) > 1 else cell_type
    dataset = sys.argv[2] if len(sys.argv) > 2 else dataset
    main(cell_type, dataset, max_hops)

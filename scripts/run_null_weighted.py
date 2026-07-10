import pickle
import networkx as nx
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import NULL_CONNECTOMES_DIR
from connectomics.utils.extract_sensory import _get_sensory, _get_nonsensory_type_edges
from connectomics.modeling.diffusion_models import weighted_sensory_propagation
from connectomics.modeling.null_comparison import run_model_on_nulls

dataset = "fafb" # Specify which dataset to run the analysis on.

def main(dataset="fafb"):
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

    print("Running weighted_sensory_propagation on the real connectome...")
    actual = weighted_sensory_propagation(G, sensory_types)

    with open(str(NULL_CONNECTOMES_DIR / f"{dataset}_null_sensory_edges.pkl"), "rb") as file:
        null_edge_sets = pickle.load(file)

    nonsensory_edges = _get_nonsensory_type_edges(dataset)

    print(f"Running weighed_sensory_propagation over {len(null_edge_sets)} null connectomes...")
    nulls = run_model_on_nulls(
        weighted_sensory_propagation,
        null_edge_sets,
        nonsensory_edges,
        sensory_types,
    )

    actual_dir = NULL_CONNECTOMES_DIR / f"{dataset}_actual_weighted_results.pkl"
    null_dir = NULL_CONNECTOMES_DIR / f"{dataset}_null_weighted_results.pkl"
    with open(str(actual_dir), "wb") as file:
        pickle.dump(actual, file)
    with open(str(null_dir), "wb") as file:
        pickle.dump(nulls, file)
    print(f"Saved results to {str(NULL_CONNECTOMES_DIR)}")

if __name__ == "__main__":
    main(dataset)

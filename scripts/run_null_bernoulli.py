import pickle
import networkx as nx
from connectomics.processing.data_loader import DATASETS
from connectomics.config.paths import NULL_CONNECTOMES_DIR
from connectomics.utils.extract_sensory import _get_sensory, _get_nonsensory_type_edges
from connectomics.modeling.diffusion_models import bernoulli_sensory_cascade
from connectomics.modeling.null_comparison import run_model_on_nulls, run_model_repeated

dataset = "fafb" # Specify which dataset to run the analysis on.
n_repeats = 100 # Number of seeded repeats on the real connectome, for a reference distribution comparable to the null distribution.
model_kwargs = {
    "p_transmission": 0.0005, # Found to be the value that allows for finding meaningful differences among cells.
    "max_steps": 100,
    "n_trials": 100,
}

def main(dataset="fafb", n_repeats=1000, model_kwargs=None):
    model_kwargs = model_kwargs or {}
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

    print(f"Running bernoulli_sensory_cascade {n_repeats} times on the real connectome...")
    actual = run_model_repeated(
        bernoulli_sensory_cascade,
        G,
        sensory_types,
        n_repeats=n_repeats,
        model_kwargs=model_kwargs,
    )

    with open(str(NULL_CONNECTOMES_DIR / f"{dataset}_null_sensory_edges.pkl"), "rb") as file:
        null_edge_sets = pickle.load(file)

    nonsensory_edges = _get_nonsensory_type_edges(dataset)

    print(f"Running bernoulli_sensory_cascade over {len(null_edge_sets)} null connectomes...")
    nulls = run_model_on_nulls(
        bernoulli_sensory_cascade,
        null_edge_sets,
        nonsensory_edges,
        sensory_types,
        model_kwargs=model_kwargs,
    )

    actual_dir = NULL_CONNECTOMES_DIR / f"{dataset}_actual_bernoulli_results.pkl"
    null_dir = NULL_CONNECTOMES_DIR / f"{dataset}_null_bernoulli_results.pkl"
    with open(str(actual_dir), "wb") as file:
        pickle.dump(actual, file)
    with open(str(null_dir), "wb") as file:
        pickle.dump(nulls, file)
    print(f"Saved results to {str(NULL_CONNECTOMES_DIR)}")

if __name__ == "__main__":
    main(dataset, n_repeats, model_kwargs)

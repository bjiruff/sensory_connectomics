import os
from multiprocessing import Pool
import networkx as nx
import numpy as np


def build_null_graph(null_edges, nonsensory_edges):
    """
    Reconstruct a full null connectome graph from one entry of the pickled null edge list
    produced by scripts/build_null_connectomes.py, plus the constant non-sensory -> non-sensory
    edges (see connectomics.utils.extract_sensory._get_nonsensory_type_edges).

    null_edges: list of (pre_type, post_type, {"weight": syn_count}) for one null connectome,
                covering sensory -> second order edges only.
    nonsensory_edges: list of (pre_type, post_type, {"weight": syn_count}), identical across
                       every null connectome.
    """
    G = nx.DiGraph()
    G.add_edges_from(null_edges)
    G.add_edges_from(nonsensory_edges)
    return G


_worker_model_fn = None
_worker_nonsensory_edges = None
_worker_sensory_nodes_by_modality = None
_worker_model_kwargs = None
_worker_base_seed = None

def _init_null_worker(model_fn, nonsensory_edges, sensory_nodes_by_modality, model_kwargs, base_seed):
    global _worker_model_fn, _worker_nonsensory_edges, _worker_sensory_nodes_by_modality
    global _worker_model_kwargs, _worker_base_seed
    _worker_model_fn = model_fn
    _worker_nonsensory_edges = nonsensory_edges
    _worker_sensory_nodes_by_modality = sensory_nodes_by_modality
    _worker_model_kwargs = model_kwargs
    _worker_base_seed = base_seed
    os.environ["OMP_NUM_THREADS"] = "1"

def _run_model_on_one_null(args):
    idx, null_edges = args
    G = build_null_graph(null_edges, _worker_nonsensory_edges)

    kwargs = dict(_worker_model_kwargs)
    if _worker_base_seed is not None:
        kwargs["random_seed"] = _worker_base_seed + idx

    return _worker_model_fn(G, _worker_sensory_nodes_by_modality, **kwargs)


def run_model_on_nulls(
        model_fn,
        null_edge_sets,
        nonsensory_edges,
        sensory_nodes_by_modality,
        model_kwargs=None,
        processes=None,
        base_seed=None,
        ):
    """
    Run a diffusion model once per null connectome, in parallel.

    model_fn: a function with signature (G, sensory_nodes_by_modality, **kwargs) ->
              {node: {modality: value, ..., "total": value}}, e.g. weighted_sensory_propagation
              or bernoulli_sensory_cascade. For a stochastic model_fn, each call already
              represents that model's own internal averaging (e.g. bernoulli_sensory_cascade's
              n_trials) — this function does not repeat calls per null.
    null_edge_sets: list of per-null sensory -> second order edge lists, as produced by
                    scripts/build_null_connectomes.py (one null connectome per entry).
    nonsensory_edges: constant non-sensory -> non-sensory edge list, shared by every null.
    sensory_nodes_by_modality: dict of {modality_name: [list of sensory type/node names]}.
    model_kwargs: extra keyword arguments forwarded to model_fn on every call.
    processes: number of worker processes (defaults to Pool's normal os.cpu_count() behavior).
    base_seed: if model_fn accepts a `random_seed` keyword argument (e.g.
               bernoulli_sensory_cascade) and reproducible per-null seeding is desired, each
               null i is called with random_seed=base_seed + i. Leave None to let model_fn use
               its own default (non-reproducible) randomness.

    Returns: dict mapping each node to {modality: np.ndarray of length len(null_edge_sets), ...}
             (plus "total"), i.e. the model's output stacked across all null connectomes.
    """
    model_kwargs = model_kwargs or {}
    tasks = list(enumerate(null_edge_sets))

    results_by_node = None
    completed = 0

    with Pool(
        processes=processes,
        initializer=_init_null_worker,
        initargs=(model_fn, nonsensory_edges, sensory_nodes_by_modality, model_kwargs, base_seed)
    ) as pool:
        for result in pool.imap_unordered(_run_model_on_one_null, tasks):
            if results_by_node is None:
                keys = next(iter(result.values())).keys()
                results_by_node = {node: {k: [] for k in keys} for node in result}

            for node, values in result.items():
                for k, v in values.items():
                    results_by_node[node][k].append(v)

            completed += 1
            if completed % 50 == 0:
                print(f"  Finished {completed}/{len(tasks)} nulls")

    for values in results_by_node.values():
        for k in values:
            values[k] = np.array(values[k], dtype=np.float64)

    return results_by_node


_worker_repeat_model_fn = None
_worker_repeat_G = None
_worker_repeat_sensory_nodes_by_modality = None
_worker_repeat_model_kwargs = None

def _init_repeat_worker(model_fn, G, sensory_nodes_by_modality, model_kwargs):
    global _worker_repeat_model_fn, _worker_repeat_G, _worker_repeat_sensory_nodes_by_modality
    global _worker_repeat_model_kwargs
    _worker_repeat_model_fn = model_fn
    _worker_repeat_G = G
    _worker_repeat_sensory_nodes_by_modality = sensory_nodes_by_modality
    _worker_repeat_model_kwargs = model_kwargs
    os.environ["OMP_NUM_THREADS"] = "1"

def _run_model_with_seed(seed):
    kwargs = dict(_worker_repeat_model_kwargs)
    kwargs["random_seed"] = seed
    return _worker_repeat_model_fn(_worker_repeat_G, _worker_repeat_sensory_nodes_by_modality, **kwargs)


def run_model_repeated(
        model_fn,
        G,
        sensory_nodes_by_modality,
        n_repeats,
        model_kwargs=None,
        base_seed=0,
        processes=None,
        ):
    """
    Run a diffusion model repeatedly on a single fixed graph (typically the real connectome),
    to build a reference distribution comparable to run_model_on_nulls' null distribution.
    This is only meaningful for a stochastic model_fn (e.g. bernoulli_sensory_cascade) — a
    deterministic model_fn (e.g. weighted_sensory_propagation) returns the same result every call,
    so a single direct call to model_fn should be used instead of this function.

    model_fn: must accept a `random_seed` keyword argument.
    G: the graph to run the model on repeatedly (e.g. the real connectome).
    sensory_nodes_by_modality: dict of {modality_name: [list of sensory type/node names]}.
    n_repeats: number of times to call model_fn, each with a distinct seed.
    model_kwargs: extra keyword arguments forwarded to model_fn on every call.
    base_seed: seeds passed to model_fn are base_seed, base_seed + 1, ..., base_seed + n_repeats - 1.
    processes: number of worker processes (defaults to Pool's normal os.cpu_count() behavior).

    Returns: dict mapping each node to {modality: np.ndarray of length n_repeats, ...} (plus "total").
    """
    model_kwargs = model_kwargs or {}
    seeds = list(range(base_seed, base_seed + n_repeats))

    results_by_node = None

    with Pool(
        processes=processes,
        initializer=_init_repeat_worker,
        initargs=(model_fn, G, sensory_nodes_by_modality, model_kwargs)
    ) as pool:
        for result in pool.imap_unordered(_run_model_with_seed, seeds):
            if results_by_node is None:
                keys = next(iter(result.values())).keys()
                results_by_node = {node: {k: [] for k in keys} for node in result}

            for node, values in result.items():
                for k, v in values.items():
                    results_by_node[node][k].append(v)

    for values in results_by_node.values():
        for k in values:
            values[k] = np.array(values[k], dtype=np.float64)

    return results_by_node

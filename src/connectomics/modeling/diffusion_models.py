from typing import List, Dict, Optional
from collections import deque
from connectomics.processing.data_loader import DATASETS
import random
import numpy as np
from scipy import sparse

def weighted_sensory_propagation(
        G: "nx.DiGraph",
        sensory_nodes_by_modality: Dict[str, List],
        epsilon=1e-6,
        max_iter=1000
        ) -> Dict[str, Dict[str, float]]:
    """
    Deterministic diffusion of sensory influence through the connectome.

    For each modality, sensory nodes of that modality are pinned to 1.0 and all
    other sensory nodes are pinned to 0.0. At each iteration every non-sensory
    node's value is updated to the synapse-weighted average of its upstream
    neighbors' values (one step of random-walk diffusion on the input-normalised
    weight matrix). Iteration continues until the maximum per-node change falls
    below epsilon, yielding a steady-state proportion that reflects how much of
    each neuron's total synaptic drive is traceable back to each modality.

    G: directed networkx graph, edges have 'weight' attribute (synapse count).
    sensory_nodes_by_modality: dict of {modality_name: [list of sensory nodes]}
                               e.g. {"olfactory": [...], "gustatory": [...], "thermosensory": [...]}.
    epsilon: convergence threshold — stop when max per-node change < epsilon.
    max_iter: maximum iterations before halting regardless of convergence.
    """

    all_nodes = list(G.nodes())
    node_index = {v: i for i, v in enumerate(all_nodes)}
    n = len(all_nodes)

    # sensory_ids = DATASETS[dataset].classifications_df.loc[
    #     DATASETS[dataset].classifications_df["super_class"] == "sensory",
    #     "root_id"
    # ]
    # sensory_types = DATASETS[dataset].celltypes_df.loc[
    #     DATASETS[dataset].celltypes_df["root_id"].isin(sensory_ids),
    #     "primary_type"
    # ].unique()
    # all_sensory_nodes = set(s for s in sensory_types if s in G.nodes())
    all_sensory_nodes = set().union(*sensory_nodes_by_modality.values())

    # --- Build sparse weight matrix ---
    # FIX 1: Sort edges into a canonical order before building the matrix.
    # Unsorted edge iteration order can differ between platforms/Python versions,
    # causing the sparse matrix to accumulate values in different orders.
    edges = sorted(G.edges(data=True), key=lambda e: (node_index[e[1]], node_index[e[0]]))
    rows, cols, weights = [], [], []
    for u, v, d in edges:
        rows.append(node_index[v])
        cols.append(node_index[u])
        weights.append(d['weight'])

    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)

    # FIX 2: Use float64 dense array for row sums with Kahan summation
    # instead of sparse .sum(), which uses arbitrary internal accumulation order.
    def kahan_row_sums(matrix):
        """Compensated summation over each row of a sparse CSR matrix."""
        sums = np.zeros(matrix.shape[0], dtype=np.float64)
        compensations = np.zeros(matrix.shape[0], dtype=np.float64)
        cx = matrix.tocsr()
        for i in range(cx.shape[0]):
            row = cx.getrow(i)
            for val in row.data:
                y = val - compensations[i]
                t = sums[i] + y
                compensations[i] = (t - sums[i]) - y
                sums[i] = t
        return sums

    total_input = kahan_row_sums(W)

    total_input_safe = np.where(total_input > 0, total_input, 1.0)
    D_inv = sparse.diags(1.0 / total_input_safe)
    W_norm = D_inv @ W

    # FIX 3: Convert W_norm to CSR with sorted indices.
    # This ensures the sparse matvec multiplies columns in the same order
    # on every platform, which is the dominant source of compounding error.
    W_norm = W_norm.tocsr()
    W_norm.sort_indices()

    non_sensory_mask = np.array([v not in all_sensory_nodes for v in all_nodes])

    # FIX 4: Precompute other_sensory_idx outside the iteration loop.
    # Not a float fix, but avoids repeated reallocation and is more correct.
    other_sensory_by_modality = {
        modality: np.array([
            node_index[v] for mod, nodes in sensory_nodes_by_modality.items()
            if mod != modality
            for v in nodes
            if v in node_index
        ])
        for modality in sensory_nodes_by_modality
    }

    P_by_modality = {}

    for modality, sensory_nodes in sensory_nodes_by_modality.items():
        sensory_idx = np.array([node_index[v] for v in sensory_nodes if v in node_index])
        other_sensory_idx = other_sensory_by_modality[modality]

        P = np.zeros(n, dtype=np.float64)
        P[sensory_idx] = 1.0

        for iteration in range(max_iter):
            # FIX 5: Use explicit float64 matvec.
            # W_norm.dot() can sometimes dispatch to a less precise path;
            # astype ensures the output buffer stays float64.
            P_new = W_norm.dot(P).astype(np.float64)

            P_new[sensory_idx] = 1.0
            if len(other_sensory_idx) > 0:
                P_new[other_sensory_idx] = 0.0

            max_delta = np.max(np.abs(P_new[non_sensory_mask] - P[non_sensory_mask]))
            P = P_new

            if max_delta < epsilon:
                # print(f"  [{modality}] Converged after {iteration + 1} iterations")
                break
        else:
            print(f"  [{modality}] Warning: did not converge after {max_iter} iterations")

        P_by_modality[modality] = P

    modalities = list(P_by_modality.keys())
    P_matrix = np.stack([P_by_modality[m] for m in modalities], axis=0)

    # FIX 6: Kahan sum across modalities instead of plain .sum(axis=0)
    P_total = np.zeros(n, dtype=np.float64)
    compensation = np.zeros(n, dtype=np.float64)
    for m in range(len(modalities)):
        y = P_matrix[m] - compensation
        t = P_total + y
        compensation = (t - P_total) - y
        P_total = t

    # FIX 7: Round to 3 decimal places and downcast to float16 for a more
    # space-efficient result. All values lie in [0, 1], where float16's ~11-bit
    # mantissa resolves increments as fine as 0.00024-0.00049 (worst case near 1),
    # comfortably preserving 3 decimal places while quartering the per-value size
    # relative to float64.
    P_matrix = np.round(P_matrix, 4).astype(np.float16)
    P_total = np.round(P_total, 4).astype(np.float16)

    results = {
        v: {modality: P_matrix[m, node_index[v]] for m, modality in enumerate(modalities)}
          | {"total": P_total[node_index[v]]}
        for v in all_nodes
    }

    return results

def bernoulli_sensory_cascade(
        G: "nx.DiGraph",
        sensory_nodes_by_modality: Dict[str, List],
        p_transmission: float = 0.0005,
        max_steps: int = 100,
        n_trials: int = 100,
        random_seed: int = None
        ) -> Dict[str, Dict[str, float]]:
    """
    Stochastic cascade model of sensory signal propagation.
    Based on the "unimodal" model presented in the following paper:
    Betzel RF, Puxeddu MG, Seguin C, Misic B (2026) Cascades and convergence: 
    Dynamic signal flow in a synapse-level brain network. PLOS Complex Systems 3(3): e0000091. 
    https://doi.org/10.1371/journal.pcsy.0000091

    Each neuron exists in one of three states: inactive (0), active (1), or refractory (2).
    For each modality, all of its sensory neurons are activated as seeds at t=0. Active neurons
    probabilistically activate inactive post-synaptic partners at each step, then immediately
    transition to refractory (inactive → active → refractory). Results are averaged over
    n_trials Monte Carlo runs per modality.

    If neuron i makes k synapses onto j, the probability that j is activated by i in one step
    is: 1 - (1 - p_transmission)^k, representing k independent Bernoulli trials. Multiple 
    pre-synaptic inputs are treated as independent.

    G: directed networkx graph, edges have 'weight' attribute (synapse count).
    sensory_nodes_by_modality: dict of {modality_name: [list of sensory nodes]}.
    p_transmission: per-synapse transmission probability (biologically realistic range: 0.01-0.05).
    max_steps: maximum cascade steps before halting a trial.
    n_trials: number of Monte Carlo trials per modality.
    random_seed: seed for reproducibility.

    Returns: dict mapping each node to {modality: activation_probability, ..., "total": sum}.
    """
    all_nodes = list(G.nodes())
    node_index = {v: i for i, v in enumerate(all_nodes)}
    n = len(all_nodes)

    # Build sparse weight matrix W where W[v, u] = synapse count u→v
    edges = sorted(G.edges(data=True), key=lambda e: (node_index[e[1]], node_index[e[0]]))
    rows, cols, weights = [], [], []
    for u, v, d in edges:
        rows.append(node_index[v])
        cols.append(node_index[u])
        weights.append(float(d['weight']))

    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)
    W.sort_indices()

    q = 1.0 - p_transmission
    rng = np.random.default_rng(random_seed)

    activation_by_modality = {}

    for modality, sensory_nodes in sensory_nodes_by_modality.items():
        # Seed a random cell type in a sensory modality.
        # sensory_node = random.choice(sensory_nodes)
        # seed_idx = node_index[sensory_node]
        seed_idx = np.array([node_index[v] for v in sensory_nodes if v in node_index])
        activation_counts = np.zeros(n, dtype=np.float64)

        for _ in range(n_trials):
            # 0=inactive, 1=active, 2=refractory
            state = np.zeros(n, dtype=np.int8)
            state[seed_idx] = 1
            ever_active = np.zeros(n, dtype=bool)
            ever_active[seed_idx] = True

            for _ in range(max_steps):
                active_mask = (state == 1)
                if not active_mask.any():
                    break

                # Total synapse-weighted input to each node from currently active neurons:
                # w_sum[v] = sum over active u of (synapse count u→v)
                w_sum = W.dot(active_mask.astype(np.float32))

                # P(v activated) = 1 - q^(total synapses from active neurons), per-node
                prob_activate = 1.0 - np.power(q, w_sum)

                # Sample: only inactive nodes can be activated
                newly_active = (rng.random(n) < prob_activate) & (state == 0)

                # inactive→active runs before active→refractory so there is no overlap:
                # newly_active was filtered to state==0, while active_mask is state==1
                state[active_mask] = 2
                state[newly_active] = 1
                ever_active |= newly_active

            activation_counts += ever_active

        activation_by_modality[modality] = activation_counts / n_trials

    modalities = list(activation_by_modality.keys())
    P_matrix = np.stack([activation_by_modality[m] for m in modalities], axis=0)
    # P_total = P_matrix.sum(axis=0)

    # results = {
    #     v: {modality: P_matrix[m, node_index[v]] for m, modality in enumerate(modalities)}
    #        | {"total": P_total[node_index[v]]}
    #     for v in all_nodes
    # }
    results = {
        v: {modality: P_matrix[m, node_index[v]] for m, modality in enumerate(modalities)}
        for v in all_nodes
    }

    return results

def sensory_orders(
        G: "nx.DiGraph",
        sensory_nodes_by_modality: Dict[str, List]
        ) -> Dict[str, Dict[str, Optional[int]]]:
    """
    BFS-based assignment of synaptic order relative to each sensory modality.

    For each modality, its sensory neurons are assigned order 1. Their direct
    post-synaptic targets receive order 2, those targets' downstream receive
    order 3, and so on. First assignment wins. Once a neuron has an order it
    cannot be reassigned a higher one. Neurons unreachable from a modality's
    sensory neurons receive order None.

    G: directed networkx graph (edges point pre → post-synaptic).
    sensory_nodes_by_modality: dict of {modality_name: [list of sensory nodes]}.

    Returns: dict mapping each node to {modality: order_or_None, ...}.
    """
    all_nodes = list(G.nodes())
    results = {v: {} for v in all_nodes}

    for modality, sensory_nodes in sensory_nodes_by_modality.items():
        order = {}
        queue = deque()

        for v in sensory_nodes:
            if v in G:
                order[v] = 1
                queue.append(v)

        while queue:
            node = queue.popleft()
            next_order = order[node] + 1
            for successor in G.successors(node):
                if successor not in order:
                    order[successor] = next_order
                    queue.append(successor)

        for v in all_nodes:
            results[v][modality] = order.get(v, None)

    return results

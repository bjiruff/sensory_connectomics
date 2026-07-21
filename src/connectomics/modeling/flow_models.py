from typing import Dict, List
from collections import deque
import networkx as nx


def _multi_source_hop_distances(G: "nx.DiGraph", sources, cutoff: int) -> Dict:
    """
    Unweighted (edge-count) hop distance from the nearest node in `sources`, via BFS
    capped at `cutoff` hops. Nodes not reachable within `cutoff` hops are absent from
    the returned dict.
    """
    dist = {s: 0 for s in sources if s in G}
    queue = deque(dist)
    while queue:
        u = queue.popleft()
        if dist[u] == cutoff:
            continue
        for v in G.successors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def maxflow_sensory_analysis(
        G: "nx.DiGraph",
        sensory_nodes_by_modality: Dict[str, List],
        target,
        max_hops: int = 4,
        capacity: str = "weight",
        ) -> Dict[str, Dict[str, float]]:
    """
    Max synaptic flow from each sensory modality to `target`, restricted to directed
    paths of at most `max_hops` edges (i.e. paths of length 1, 2, ..., max_hops hops
    all count; a direct sensory->target edge counts as a 1-hop path).

    Path-length-bounded max flow is not the same as running ordinary max flow on the
    subgraph of edges that happen to lie on some short simple path: that subgraph can
    still carry flow along paths longer than max_hops by chaining edges pulled in from
    different enumerated paths. Instead this builds a time-expanded ("layered") graph:
    max_hops + 1 copies of each candidate node, one per hop count, with an edge
    (u, i) -> (v, i+1) for every original edge u->v. Every unit of flow through this
    expanded graph corresponds to a walk of length exactly its final layer index in the
    original graph, so bounding the layers to 0..max_hops exactly bounds path length.
    A super-source feeds each modality's sensory nodes at layer 0 (infinite capacity,
    so the source is never the bottleneck), and `target` drains to a super-sink from
    every layer 1..max_hops (so any hop count in that range counts). `target` itself
    has no outgoing edges in the expanded graph, so a path ends the first time it
    reaches `target` rather than possibly looping back through it later.

    Sensory nodes belonging to OTHER modalities are excluded from the candidate node
    set entirely, so a path can't count as e.g. olfactory input to `target` if it
    actually routes through a gustatory sensory neuron partway through (mirrors the
    pinning-to-0 treatment of other-modality sensory nodes in weighted_sensory_propagation).

    G: directed networkx graph, edges have a `capacity` attribute (default 'weight',
       synapse count).
    sensory_nodes_by_modality: dict of {modality_name: [list of sensory nodes]}.
    target: the node (e.g. cell type) to measure flow into.
    max_hops: maximum path length in edges to allow.
    capacity: edge attribute name to use as flow capacity.

    Returns: {target: {modality: max_flow_value, ..., "total": sum across modalities}}.
    """
    if target not in G:
        raise ValueError(f"Target node '{target}' not found in graph.")

    all_sensory_nodes = set().union(*sensory_nodes_by_modality.values())
    flows = {}

    for modality, raw_sources in sensory_nodes_by_modality.items():
        sources = [s for s in raw_sources if s in G]
        other_sensory = all_sensory_nodes - set(sources)

        if not sources or target in other_sensory:
            flows[modality] = 0.0
            continue

        usable_nodes = set(G.nodes()) - other_sensory
        G_usable = G.subgraph(usable_nodes)

        fwd = _multi_source_hop_distances(G_usable, sources, max_hops)
        bwd = _multi_source_hop_distances(G_usable.reverse(copy=False), [target], max_hops)

        if target not in fwd or fwd[target] == 0:
            # Target unreachable within max_hops, or target is itself a source (0 hops doesn't count).
            flows[modality] = 0.0
            continue

        candidates = {
            v for v in fwd.keys() & bwd.keys()
            if fwd[v] + bwd[v] <= max_hops
        }

        SOURCE, SINK = "__source__", "__sink__"
        H = nx.DiGraph()
        for s in sources:
            if s in candidates:
                H.add_edge(SOURCE, (s, 0), capacity=float("inf"))
        for layer in range(1, max_hops + 1):
            H.add_edge((target, layer), SINK, capacity=float("inf"))

        for u, v, data in G_usable.edges(data=True):
            if u == target or u not in candidates or v not in candidates:
                continue
            w = data.get(capacity, 0)
            if w <= 0:
                continue
            for layer in range(max_hops):
                H.add_edge((u, layer), (v, layer + 1), capacity=w)

        flows[modality] = nx.maximum_flow_value(H, SOURCE, SINK, capacity="capacity")

    return {target: {**flows, "total": sum(flows.values())}}

from typing import List, Tuple, Dict, Optional
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from connectomics.processing.data_loader import DATASETS


def build_connectivity_graph(dataset: str, mode: str = "types", threshold: int = 5) -> nx.DiGraph:
    """
    Build a directed connectivity graph for a dataset, keeping only edges with
    at least "
    threshold" synapses.

    mode: "types" builds a graph of cell types (nodes are primary_type strings,
          using the precomputed type_connections_df). "ids" builds a graph of
          individual neurons (nodes are root_ids, using cell_connections_df).
    """
    if mode == "types":
        connections_df = DATASETS[dataset].type_connections_df
        source_col, target_col = "pre_type", "post_type"
    elif mode == "ids":
        connections_df = DATASETS[dataset].cell_connections_df
        source_col, target_col = "pre_root_id", "post_root_id"
    else:
        raise ValueError("mode must be either 'ids' or 'types'.")

    edges_df = connections_df.loc[connections_df["syn_count"] >= threshold]
    # edges_df = edges_df.loc[(edges_df[source_col] != edges_df[target_col])]

    return nx.from_pandas_edgelist(
        edges_df,
        source=source_col,
        target=target_col,
        edge_attr="syn_count",
        create_using=nx.DiGraph(),
    )


def find_shortest_paths(G: nx.DiGraph, source, target) -> List[List]:
    """Return every shortest (fewest-hop) directed path from source to target."""
    if source not in G:
        raise ValueError(f"Source node '{source}' not found in graph.")
    if target not in G:
        raise ValueError(f"Target node '{target}' not found in graph.")
    return list(nx.all_shortest_paths(G, source, target))


def paths_to_subgraph(G: nx.DiGraph, paths: List[List]) -> nx.DiGraph:
    """
    Build a subgraph containing only the nodes/edges that appear on the given
    paths, with each node labeled by its distance (layer) from the source so
    it can be laid out column by column when drawn.

    Since the subgraph pulls in every edge between path nodes present in G,
    it can include edges that aren't feedforward with respect to the layer
    assignment (edges back to an earlier layer, or edges within the same
    layer). Those are dropped so only strictly feedforward edges (target
    layer > source layer) remain for drawing.
    """
    nodes = {node for path in paths for node in path}
    H = G.subgraph(nodes).copy()
    for path in paths:
        for layer, node in enumerate(path):
            H.nodes[node]["layer"] = layer

    non_feedforward = [
        (u, v) for u, v in H.edges()
        if H.nodes[v]["layer"] <= H.nodes[u]["layer"]
    ]
    H.remove_edges_from(non_feedforward)

    return H


def draw_pathways(H: nx.DiGraph, labels: Optional[Dict] = None, ax: Optional[Axes] = None) -> Axes:
    """
    Draw a shortest-path subgraph produced by `paths_to_subgraph`, arranging
    nodes into columns by their distance from the source.

    labels: optional mapping from node to display label (e.g. root_id -> cell
            type name). Nodes not present in the mapping are labeled with
            their own name.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    pos = nx.multipartite_layout(H, subset_key="layer")
    edge_labels = nx.get_edge_attributes(H, "syn_count")
    node_labels = {n: (labels or {}).get(n, n) for n in H.nodes()}

    nx.draw_networkx_nodes(H, pos, node_size=700, ax=ax)
    nx.draw_networkx_edges(H, pos, width=2, ax=ax)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, ax=ax)
    nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=10, font_family="sans-serif", ax=ax)

    ax.axis("off")
    return ax


def find_and_draw_shortest_paths(
        source,
        target,
        dataset: str = "fafb",
        mode: str = "types",
        threshold: int = 5,
        labels: Optional[Dict] = None,
        ax: Optional[Axes] = None
        ) -> Tuple[List[List], nx.DiGraph]:
    """Build the connectivity graph, find all shortest paths, and draw them."""
    G = build_connectivity_graph(dataset, mode=mode, threshold=threshold)
    paths = find_shortest_paths(G, source, target)
    H = paths_to_subgraph(G, paths)
    draw_pathways(H, labels=labels, ax=ax)
    return paths, H

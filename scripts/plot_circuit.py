import sys
import numpy as np
import networkx as nx
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from collections import namedtuple
from connectomics.config.paths import FIGS_DIR
from connectomics.processing.data_loader import DATASETS


def _consensus_nt(group):
    counts = group.value_counts()
    if len(counts) == 0:
        return None
    if counts.iloc[0] / len(group) >= 0.5:
        return counts.index[0]
    return None

def _get_edge_drawing_data(G: nx.DiGraph, colormap: Dict, min_width=0.5, max_width=8.0):
    # Get edge data for feedforward and reciprocal edges for drawing purposes.

    EdgeData = namedtuple("EdgeData", ["edges", "weights", "widths", "colors"])

    edgeset = set(G.edges())
    
    fwd_edges, fwd_colors, fwd_weights = [], [], []
    rec_edges, rec_colors, rec_weights = [], [], []
    for (u, v, d) in G.edges(data=True):
        if (v, u) in edgeset:
            rec_edges.append((u, v))
            rec_colors.append(colormap.get(G.nodes[u]["nt_type"], colormap["Unknown"]))
            rec_weights.append(d["weight"])
        else:
            fwd_edges.append((u, v))
            fwd_colors.append(colormap.get(G.nodes[u]["nt_type"], colormap["Unknown"]))
            fwd_weights.append(d["weight"])
    
    # Get weights/synapse counts of feedforward and reciprocal edges so we can set their widths right afterwards.
    # fwd_weights = np.array([G[u][v]["weight"] for (u, v) in fwd_edges])
    # rec_weights = np.array([G[u][v]["weight"] for (u, v) in rec_edges])
    all_weights = np.concatenate((fwd_weights, rec_weights))

    # Set all weights together to ensure that widths are all consistent with each other.
    all_widths = _set_edge_widths(all_weights, min_width, max_width)
    fwd_widths = all_widths[:len(fwd_weights)]
    rec_widths = all_widths[len(fwd_weights):]
    
    fwd_data = EdgeData(fwd_edges, fwd_weights, fwd_widths, fwd_colors)
    rec_data = EdgeData(rec_edges, rec_weights, rec_widths, rec_colors)

    return fwd_data, rec_data


# Set width of an edge based on log synapse count.
def _set_edge_widths(weights: list, min_width=0.5, max_width=8.0):
    log_weights = np.log(weights)
    if log_weights.max() > log_weights.min():
        widths = min_width + (log_weights - log_weights.min()) / (
            log_weights.max() - log_weights.min()
        ) * (max_width - min_width)
    else:
        widths = np.full_like(log_weights, min_width)
    return widths


def main(
    neurons: list, 
    ax: Axes, 
    node_attrs: Dict, 
    edge_attrs: Dict, 
    dataset="fafb", 
    mode="types", 
    io=True
    ):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    neurotransmitters_df = DATASETS[dataset].neurotransmitters_df
    celltypes_df = DATASETS[dataset].celltypes_df
    cell_connections_df = DATASETS[dataset].cell_connections_df
    type_connections_df = DATASETS[dataset].type_connections_df

    
    id_to_type = celltypes_df.set_index("root_id")["primary_type"]

    nt_types = neurotransmitters_df["nt_type"].unique()
    colors = plt.colormaps["tab10"].resampled(len(nt_types))
    nt_colormap = {nt: colors(i) for i, nt in enumerate(nt_types)}
    nt_colormap["Unknown"] = nt_colormap.pop(np.nan)

    if mode == "ids":
        connections_df = cell_connections_df.copy()
        connections_df.rename(columns={"pre_root_id": "pre", "post_root_id": "post", "syn_count": "weight"}, inplace=True)
        cell_to_nt = neurotransmitters_df.set_index("root_id")["nt_type"]
    elif mode == "types":
        connections_df = type_connections_df.copy()
        connections_df.rename(columns={"pre_type": "pre", "post_type": "post", "syn_count": "weight"}, inplace=True)
        neurotransmitters_df["primary_type"] = neurotransmitters_df["root_id"].map(id_to_type)
        cell_to_nt = neurotransmitters_df.groupby("primary_type")["nt_type"].apply(_consensus_nt)

    # Get connectivity of specified neurons among themselves.
    # Do not include self-loops.
    connections_df = connections_df.loc[
        (connections_df["pre"].isin(neurons)) &
        (connections_df["post"].isin(neurons)) &
        (connections_df["pre"] != connections_df["post"])
    ]
    # edges = [
    #     (row.pre_root_id, row.post_root_id, {"weight": row.syn_count})
    #     for row in connections_df.itertuples()
    # ]

    G = nx.from_pandas_edgelist(
        connections_df,
        source="pre",
        target="post",
        edge_attr="weight",
        create_using=nx.DiGraph()
    )
    for node in G.nodes():
        G.nodes[node]["nt_type"] = cell_to_nt[node]

    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")

    node_colors = [nt_colormap.get(G.nodes[node]["nt_type"], nt_colormap["Unknown"]) for node in G.nodes()]


    nx.draw_networkx_nodes(
        G, 
        pos=pos, 
        node_color=node_colors, 
        ax=ax,
        **node_attrs,
    )

    G_fwd, G_rec = _get_edge_drawing_data(G, nt_colormap, 0.5, 8.0)
    # draw_networkx_nodes(G, pos, nodelist=None, node_size=300, node_color='#1f78b4', node_shape='o', alpha=None, cmap=None, vmin=None, vmax=None, ax=None, linewidths=None, edgecolors=None, label=None, margins=None, hide_ticks=True)
    # Draw one-way edges as straight. edge_color='k'

    # Draw the edges twice as a workaround for the fact that when you set a thickness for networkx edge, the edge doesn't touch the post node for some reason.
    # Arrow that actually reaches the post node and supplies the arrowhead, but has no thickness.
    nx.draw_networkx_edges(
        G, 
        pos=pos, 
        edgelist=G_fwd.edges, 
        width=0, 
        edge_color=G_fwd.colors, 
        ax=ax,
        arrowsize=list(G_fwd.widths * 5 + 3),
        node_size=node_attrs["node_size"],
        connectionstyle="arc3",
        **edge_attrs,
    )
    
    # Arrow that provides the thickness of the edge.
    nx.draw_networkx_edges(
        G, 
        pos=pos, 
        edgelist=G_fwd.edges, 
        width=G_fwd.widths, 
        edge_color=G_fwd.colors,
        ax=ax,
        arrowsize=0.00001,
        node_size=node_attrs["node_size"],
        connectionstyle="arc3",
        **edge_attrs,
    )
    
    # Draw two-way edges as curved.
    # Draw them twice to get around the bug of wide edges not connecting to 
    nx.draw_networkx_edges(
        G, 
        pos=pos, 
        edgelist=G_rec.edges, 
        width=0, 
        edge_color=G_rec.colors, 
        # alpha=0.8,
        ax=ax,
        # arrows=True, 
        # arrowstyle="-|>",
        arrowsize=list(G_rec.widths * 5 + 3),
        node_size=node_attrs["node_size"],
        connectionstyle="arc3,rad=0.1",
        **edge_attrs,
    )
    
    nx.draw_networkx_edges(
        G, 
        pos=pos, 
        edgelist=G_rec.edges, 
        width=G_rec.widths, 
        edge_color=G_rec.colors,
        # alpha=0.8,
        ax=ax,
        # arrows=True, 
        # arrowstyle="-|>",
        arrowsize=0.00001,
        node_size=node_attrs["node_size"],
        connectionstyle="arc3,rad=0.1",
        **edge_attrs,
    )

        
    node_names = {n: n for n in G.nodes()}
    # These lines was for case when we would want to call a node a different name from its actual identifier in database.
    # for k, v in node_labels.items():
    #     node_names[k] = v
        
    
    def _get_io(node, mode):
        if mode == "ids":
            nonlocal cell_connections_df
            tot_in = cell_connections_df.loc[cell_connections_df["post_root_id"] == node, "syn_count"].sum()
            tot_out = cell_connections_df.loc[cell_connections_df["pre_root_id"] == node, "syn_count"].sum()
        elif mode == "types":
            nonlocal type_connections_df
            tot_in = type_connections_df.loc[type_connections_df["post_type"] == node, "syn_count"].sum()
            tot_out = type_connections_df.loc[type_connections_df["pre_type"] == node, "syn_count"].sum()
        return tot_in, tot_out
    
    if io:
        # Write total input and output synapses on nodes.
        nx.draw_networkx_labels(
            G, 
            pos=pos, 
            labels=node_names, 
            font_size=12, 
            font_color="k",
            font_weight="bold", 
            verticalalignment="bottom",
            ax=ax,
        )
        for node in G.nodes():
            tot_in, tot_out = _get_io(node, mode)
            plt.text(
                x=pos[node][0], 
                y=pos[node][1],
                s=f"I:{tot_in}\nO:{tot_out}",
                fontsize=8,
                ha="center", 
                va="top"
            )
        # Write synapse counts on each edge.
        for color in nt_colormap.values():
            fwd_ind = [i for i, value in enumerate(G_fwd.colors) if value == color]
            rec_ind = [i for i, value in enumerate(G_rec.colors) if value == color]
            # Write synapse counts on forward edges.
            nx.draw_networkx_edge_labels(
                G, 
                pos=pos, 
                edge_labels={G_fwd.edges[i]: G_fwd.weights[i] for i in fwd_ind},
                font_size=8, 
                alpha=1,
                bbox={"boxstyle": "round", "ec": color, "fc": (1.0, 1.0, 1.0), "alpha": 0.75},
                ax=ax,
                node_size=node_attrs["node_size"],
                connectionstyle="arc3",
            )
            # Write synapse counts on reciprocal edges.
            nx.draw_networkx_edge_labels(
                G, 
                pos=pos, 
                edge_labels={G_rec.edges[i]: G_rec.weights[i] for i in rec_ind},
                font_size=8, 
                alpha=1,
                bbox={"boxstyle": "round", "ec": color, "fc": (1.0, 1.0, 1.0), "alpha": 0.75},
                ax=ax,
                node_size=node_attrs["node_size"],
                connectionstyle="arc3,rad=0.1",
            )
    # Just draw node names on nodes, without synapse count information.
    else:
        nx.draw_networkx_labels(
            G, 
            pos=pos, 
            labels=node_names, 
            font_size=12, 
            font_color="k",
            font_weight="bold", 
            verticalalignment="center",
            ax=ax,
        )

    def line2d(label, color):
        res = Line2D(
                [0], [0],
                marker='o',
                color='none',
                label=label,
                markerfacecolor=color,
                markersize=8
            )
        return res
    
    legend_elements = [line2d(nt, color) for nt, color in nt_colormap.items()]
    ax.legend(handles=legend_elements, title="NT prediction")
    
    return ax


if __name__ == "__main__":
    # Specify which neurons you want to plot. Can either be root IDs or cell type names.
    neurons = ["pC1d", "pC1e", "CB2258", "TuTuAb",
               "TuTuAa", "SMP054", "cL14", "LC10a",
               "LC10b", "LC10c", "LC10d", "Li01",
               "AOTU015a", "AOTU015b", "AOTU023", "oviIN"]
    io = False # Specify whether to draw edge weights (synapses).
    mode = "types"
    dataset = "fafb" # Specify which dataset to read from.
    
    if mode == "ids":
        invalid = set(neurons) - set(DATASETS[dataset].classifications_df["root_id"])
    elif mode == "types":
        invalid = set(neurons) - set(DATASETS[dataset].celltypes_df["primary_type"])
    else:
        raise ValueError("mode must be either 'ids' or 'types'.")
    if invalid:
        print(f"Warning, specified cells do not exist in '{dataset}':\n {invalid}")

    node_attrs = {
        "node_size": 1750,
        "alpha": 0.7,
        "linewidths": 1,
        "node_shape": "o",
        "edgecolors": "black",
    }
    edge_attrs = {
        "alpha": 0.8,
        "arrows": True,
        "arrowstyle": "-|>",
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    ax = main(
        neurons=neurons,
        ax=ax,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        dataset=dataset,
        mode=mode,
        io=io,
        )

    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "circuit.png"), dpi=300)
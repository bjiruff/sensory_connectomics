import pandas as pd
from collections import defaultdict
from connectomics.processing.data_loader import DATASETS
from connectomics.config.consts import MODALITIES


def cells_per_order(G, orders):
    """
    orders: output of sensory_orders — {node: {modality: order_or_None, ...}, ...}

    Returns a long-format DataFrame with columns: order, modality, count.
    Each row is the number of cells at a given synaptic order for one modality.
    """
    counts = defaultdict(lambda: defaultdict(int))

    for node_orders in orders.values():
        for modality, order in node_orders.items():
            if order is not None:
                counts[modality][order] += 1

    rows = [
        {"order": order, "modality": modality, "count": count}
        for modality, order_counts in counts.items()
        for order, count in order_counts.items()
    ]

    return pd.DataFrame(rows).sort_values(["modality", "order"]).reset_index(drop=True)


def synapse_flow_per_order(G, orders, modality, max_order):
    """
    For a given modality, computes the proportion of output synapses from nodes
    at each order (1..max_order) classified as feedforward, lateral, or feedback.

    - feedforward: target order > source order
    - lateral:     target order == source order
    - feedback:    target order < source order

    Synapses where either endpoint has no order assignment for the modality are
    excluded. Edge weight (synapse count) is used when present; defaults to 1.

    Returns a DataFrame with columns: order, feedforward, lateral, feedback,
    where the three proportion columns sum to 1 per row.
    """
    counts = defaultdict(lambda: {"feedforward": 0, "lateral": 0, "feedback": 0})

    for u in G.nodes():
        src_order = orders.get(u, {}).get(modality)
        if src_order is None or src_order > max_order:
            continue

        for v, data in G[u].items():
            tgt_order = orders.get(v, {}).get(modality)
            if tgt_order is None:
                continue

            weight = data.get("weight", 1)
            if tgt_order > src_order:
                counts[src_order]["feedforward"] += weight
            elif tgt_order == src_order:
                counts[src_order]["lateral"] += weight
            elif tgt_order < src_order:
                counts[src_order]["feedback"] += weight

    rows = []
    for order in sorted(counts.keys()):
        c = counts[order]
        total = c["feedforward"] + c["lateral"] + c["feedback"]
        if total == 0:
            rows.append({"order": order, "feedforward": 0.0, "lateral": 0.0, "feedback": 0.0})
        else:
            rows.append({
                "order": order,
                "feedforward": c["feedforward"] / total,
                "lateral":     c["lateral"]     / total,
                "feedback":    c["feedback"]    / total,
            })

    return pd.DataFrame(rows)

def similarity_per_order(orders, order_num):
    """
    For a given order across all modalities, compute the pairwise Jaccard similarity
    of the neuron sets between each pair of modalities.

    orders: output of sensory_orders — {node: {modality: order_or_None, ...}, ...}

    Returns a square DataFrame (modalities x modalities) of Jaccard similarity values.
    """
    modality_sets = defaultdict(set)

    for node, node_orders in orders.items():
        for modality, order in node_orders.items():
            if order == order_num:
                modality_sets[modality].add(node)

    modalities = sorted(modality_sets.keys())
    n = len(modalities)
    matrix = [[0.0] * n for _ in range(n)]

    # Calculate Jaccard similarity.
    for i, m1 in enumerate(modalities):
        for j, m2 in enumerate(modalities):
            s1, s2 = modality_sets[m1], modality_sets[m2]
            union = s1 | s2
            matrix[i][j] = len(s1 & s2) / len(union) if union else 0.0

    return pd.DataFrame(matrix, index=modalities, columns=modalities)


def cells_per_class_within_order(orders, order_num, classification, dataset, mode):
    """
    For a given order (order_num) across all modalities, find how many cells 
    belong to each class annotation.

    Returns a DataFrame with columns: modality, total, cell classes
    """
    classifications_df = DATASETS[dataset].classifications_df
    celltypes_df = DATASETS[dataset].celltypes_df

    if mode not in ["id", "type"]:
        raise ValueError("mode must be 'id' or 'type'")
    if classification not in classifications_df:
        raise ValueError("classification does not exist in classifications_df")

    classes = list(classifications_df[classification].unique().dropna())
    classes.sort()
    # classes.insert(0, "total_")
    classes.append("unknown_")
    counts = {class_: 0 for class_ in classes}
    mod_counts = {modality: counts.copy() for modality in MODALITIES}

    cell_to_class = classifications_df.set_index("root_id")[classification]

    for cell, node_orders in orders.items():
        if mode == "id":
            id = cell
        elif mode == "type":
            # Assume all cells in the type are of the same classification, so just take the first id.
            id = celltypes_df.loc[celltypes_df["primary_type"] == cell, "root_id"].values[0]
        try:
            cell_class = cell_to_class[id]
        except:
            print(f"Warning: '{cell}' does not exist in dataset.")
        if cell_class is None or pd.isna(cell_class):
            cell_class = "unknown_"
        for modality, order in node_orders.items():
            if order == order_num:
                mod_counts[modality][cell_class] += 1
                # mod_counts[modality]["total_"] += 1
        
    return pd.DataFrame(mod_counts).transpose()[classes]



import matplotlib.pyplot as plt
from connectomics.config.paths import FIGS_DIR
from connectomics.utils.pathfinder import find_and_draw_shortest_paths


def main(source, target, dataset="fafb", mode="types", threshold=5):
    paths, H = find_and_draw_shortest_paths(
        source=source,
        target=target,
        dataset=dataset,
        mode=mode,
        threshold=threshold,
    )

    print(f"Found {len(paths)} shortest path(s) from '{source}' to '{target}':")
    for path in paths:
        print("  " + " -> ".join(str(node) for node in path))

    return paths, H


if __name__ == "__main__":
    source = "SMP142"       # Source cell type (or root_id if mode="ids").
    target = "LC10a"    # Target cell type (or root_id if mode="ids").
    dataset = "fafb"    # Which dataset to read from.
    mode = "types"      # "types" or "ids".
    threshold = 5       # Minimum synapse count for an edge to be considered.

    main(source, target, dataset=dataset, mode=mode, threshold=threshold)

    plt.tight_layout()
    plt.savefig(str(FIGS_DIR / "shortest_paths.png"), dpi=300)
    plt.show()

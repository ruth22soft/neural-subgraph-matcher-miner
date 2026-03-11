"""
Phase 2: Heterogeneous Synthetic Graph Generator
Extends combined_syn.py generators to produce graphs with
node types and edge types for biological/biomedical networks.
"""

import numpy as np
import networkx as nx
from common.combined_syn import (
    ERGenerator, WSGenerator, BAGenerator, PowerLawClusterGenerator
)
import deepsnap.dataset as dataset

# Biology-inspired node and edge type vocabularies
NODE_TYPES = ["Gene", "Protein", "Disease", "Drug", "Pathway", "Function"]
EDGE_TYPES = ["regulates", "binds", "inhibits", "activates",
              "associated_with", "treats", "encodes", "has_function"]

# Which edge types are valid between which node type pairs
VALID_EDGES = {
    ("Gene",    "Gene"):     ["regulates"],
    ("Gene",    "Protein"):  ["encodes"],
    ("Protein", "Protein"):  ["binds", "inhibits", "activates"],
    ("Protein", "Disease"):  ["associated_with"],
    ("Drug",    "Protein"):  ["inhibits", "activates", "binds"],
    ("Drug",    "Disease"):  ["treats"],
    ("Gene",    "Pathway"):  ["has_function"],
    ("Protein", "Function"): ["has_function"],
}


def _assign_node_types(graph, node_types=None):
    """Assign a node type (label) to every node in the graph."""
    types = node_types if node_types else NODE_TYPES
    for node in graph.nodes():
        graph.nodes[node]["label"] = np.random.choice(types)
    return graph


def _assign_edge_types(graph, edge_types=None):
    """Assign an edge type to every edge, respecting node type pairs."""
    types = edge_types if edge_types else EDGE_TYPES
    for u, v in graph.edges():
        u_type = graph.nodes[u].get("label", "Gene")
        v_type = graph.nodes[v].get("label", "Gene")
        valid = VALID_EDGES.get(
            (u_type, v_type),
            VALID_EDGES.get((v_type, u_type), types)
        )
        graph.edges[u, v]["type"] = np.random.choice(valid)
    return graph


def make_heterogeneous(graph, node_types=None, edge_types=None):
    """
    Take any homogeneous NetworkX graph and add
    node type labels and edge type labels to it.
    """
    graph = _assign_node_types(graph, node_types)
    graph = _assign_edge_types(graph, edge_types)
    return graph


class HeterogeneousERGenerator(ERGenerator):
    """ER random graph with biological node/edge type labels."""

    def __init__(self, sizes, node_types=None, edge_types=None,
                 p_alpha=1.3, **kwargs):
        super().__init__(sizes, p_alpha=p_alpha, **kwargs)
        self.node_types = node_types or NODE_TYPES
        self.edge_types = edge_types or EDGE_TYPES

    def generate(self, size=None):
        graph = super().generate(size)
        return make_heterogeneous(graph, self.node_types, self.edge_types)


class HeterogeneousWSGenerator(WSGenerator):
    """Watts-Strogatz graph with biological node/edge type labels."""

    def __init__(self, sizes, node_types=None, edge_types=None, **kwargs):
        super().__init__(sizes, **kwargs)
        self.node_types = node_types or NODE_TYPES
        self.edge_types = edge_types or EDGE_TYPES

    def generate(self, size=None):
        graph = super().generate(size)
        return make_heterogeneous(graph, self.node_types, self.edge_types)


class HeterogeneousBAGenerator(BAGenerator):
    """Barabasi-Albert graph with biological node/edge type labels."""

    def __init__(self, sizes, node_types=None, edge_types=None, **kwargs):
        super().__init__(sizes, **kwargs)
        self.node_types = node_types or NODE_TYPES
        self.edge_types = edge_types or EDGE_TYPES

    def generate(self, size=None):
        graph = super().generate(size)
        return make_heterogeneous(graph, self.node_types, self.edge_types)


class HeterogeneousPowerLawGenerator(PowerLawClusterGenerator):
    """PowerLaw cluster graph with biological node/edge type labels."""

    def __init__(self, sizes, node_types=None, edge_types=None, **kwargs):
        super().__init__(sizes, **kwargs)
        self.node_types = node_types or NODE_TYPES
        self.edge_types = edge_types or EDGE_TYPES

    def generate(self, size=None):
        graph = super().generate(size)
        return make_heterogeneous(graph, self.node_types, self.edge_types)


def get_heterogeneous_generator(sizes, size_prob=None,
                                dataset_len=None,
                                node_types=None,
                                edge_types=None):
    """
    Build an ensemble of all 4 heterogeneous generators.
    Drop-in replacement for combined_syn.get_generator().
    """
    return dataset.EnsembleGenerator(
        [
            HeterogeneousERGenerator(
                sizes, node_types=node_types,
                edge_types=edge_types, size_prob=size_prob),
            HeterogeneousWSGenerator(
                sizes, node_types=node_types,
                edge_types=edge_types, size_prob=size_prob),
            HeterogeneousBAGenerator(
                sizes, node_types=node_types,
                edge_types=edge_types, size_prob=size_prob),
            HeterogeneousPowerLawGenerator(
                sizes, node_types=node_types,
                edge_types=edge_types, size_prob=size_prob),
        ],
        dataset_len=dataset_len
    )


def get_heterogeneous_dataset(task, dataset_len, sizes,
                               size_prob=None,
                               node_types=None,
                               edge_types=None,
                               **kwargs):
    """
    Build a GraphDataset using heterogeneous generators.
    Drop-in replacement for combined_syn.get_dataset().
    """
    generator = get_heterogeneous_generator(
        sizes, size_prob=size_prob,
        dataset_len=dataset_len,
        node_types=node_types,
        edge_types=edge_types
    )
    return dataset.GraphDataset(
        None, task=task, generator=generator, **kwargs)


def main():
    sizes = np.arange(6, 31)
    ds = get_heterogeneous_dataset("graph", dataset_len=5, sizes=sizes)
    print("Heterogeneous dataset length:", len(ds))
    example = ds[0]
    g = example.G
    print("Nodes with types:")
    for n, data in list(g.nodes(data=True))[:5]:
        print(f"  Node {n}: {data.get('label')}")
    print("Edges with types:")
    for u, v, data in list(g.edges(data=True))[:5]:
        print(f"  {u}->{v}: {data.get('type')}")


if __name__ == "__main__":
    main()
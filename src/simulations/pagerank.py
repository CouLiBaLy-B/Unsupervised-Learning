"""PageRank simulation module.

Implements PageRank algorithm using Markov chain simulation on directed graphs.
"""

from typing import List

import networkx as nx
import numpy as np
import pandas as pd


class PageRankSimulator:
    """Simulates PageRank on a directed random graph.

    Args:
        num_nodes: Number of nodes in the graph (default: 8)
        edge_probability: Probability of edge creation (default: 0.5)
        social_networks: List of node names (default: first 11 common networks)
        seed: Random seed for reproducibility (default: 20222023)
    """

    DEFAULT_NETWORKS = [
        "Facebook",
        "Whatsapp",
        "Instagram",
        "Twitter",
        "Teams",
        "Discord",
        "LinkedIn",
        "Snap",
        "Bloomberg",
        "Slack",
        "Github",
    ]

    def __init__(
        self,
        num_nodes: int = 8,
        edge_probability: float = 0.5,
        social_networks: List[str] | None = None,
        seed: int = 20222023,
    ) -> None:
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.social_networks = social_networks or self.DEFAULT_NETWORKS
        self.seed = seed

        self.graph: nx.DiGraph | None = None
        self.adjacency_matrix: np.ndarray | None = None
        self.transition_matrix: np.ndarray | None = None
        self.stationary_probabilities: pd.Series | None = None

    def generate_graph(self) -> nx.DiGraph:
        """Generate a directed random graph using G(n, p) model.

        Returns:
            A networkx DiGraph object
        """
        self.graph = nx.gnp_random_graph(
            self.num_nodes,
            self.edge_probability,
            directed=True,
            seed=self.seed,
        )
        return self.graph

    def compute_adjacency_matrix(self) -> np.ndarray:
        """Compute the adjacency matrix from the generated graph.

        Returns:
            Adjacency matrix as numpy array
        """
        if self.graph is None:
            self.generate_graph()

        mat = nx.adjacency_matrix(self.graph)
        self.adjacency_matrix = np.array(mat.todense(), dtype=int)
        return self.adjacency_matrix

    def compute_transition_matrix(self, alpha: float = 0.85) -> np.ndarray:
        """Compute the transition matrix using the PageRank damping factor.

        Args:
            alpha: Damping factor (default: 0.85)

        Returns:
            Transition matrix as numpy array
        """
        if self.adjacency_matrix is None:
            self.compute_adjacency_matrix()

        n = self.num_nodes
        # Initialize P as the normalized adjacency matrix
        P = self.adjacency_matrix.astype(float)
        row_sums = P.sum(axis=1)

        # Handle dangling nodes (rows with all zeros) by distributing probability uniformly
        # For other nodes, normalize the row
        for i in range(n):
            if row_sums[i] == 0:
                P[i, :] = 1.0 / n
            else:
                P[i, :] /= row_sums[i]

        # Google Matrix M = alpha * P + (1 - alpha) * (1/n * ones_matrix)
        ones_matrix = np.ones((n, n)) / n
        self.transition_matrix = alpha * P + (1 - alpha) * ones_matrix

        return self.transition_matrix

    def compute_stationary_probability(
        self,
        power: int = 1000,
        alpha: float = 0.85,
    ) -> pd.Series:
        """Compute stationary probability using matrix power iteration.

        Args:
            power: Number of iterations (default: 1000)
            alpha: Damping factor (default: 0.85)

        Returns:
            Stationary probabilities as pandas Series
        """
        if self.transition_matrix is None:
            self.compute_transition_matrix(alpha)

        transition_power = np.linalg.matrix_power(self.transition_matrix, power)
        stationary = transition_power[0]  # First row converges to stationary dist

        self.stationary_probabilities = pd.Series(
            stationary,
            index=self.social_networks[: self.num_nodes],
            name="Stationary Probability",
        )
        return self.stationary_probabilities

    def simulate_markov_chain(self, length: int = 1000) -> List[str]:
        """Simulate a Markov chain walk through the graph.

        Args:
            length: Number of steps in the chain (default: 1000)

        Returns:
            List of visited node names
        """
        if self.transition_matrix is None:
            raise ValueError(
                "Transition matrix not computed. Call compute_transition_matrix() first."
            )

        networks = self.social_networks[: self.num_nodes]
        # Pre-build index map to avoid O(n) list.index() on every step
        name_to_idx = {name: idx for idx, name in enumerate(networks)}
        current_state = np.random.choice(networks)
        chain = [current_state]

        for _ in range(length):
            current_idx = name_to_idx[current_state]
            current_state = np.random.choice(
                networks,
                p=self.transition_matrix[current_idx],
            )
            chain.append(current_state)

        return chain

    def get_transition_string(self, length: int = 1000) -> str:
        """Get the Markov chain chain as a string representation.

        Args:
            length: Number of steps (default: 1000)

        Returns:
            String like "Facebook -> Twitter -> Instagram -> ..."
        """
        chain = self.simulate_markov_chain(length)
        return " -> ".join(chain)

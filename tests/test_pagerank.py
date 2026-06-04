"""Tests for the PageRank simulator."""

import numpy as np
import pytest

from src.simulations.pagerank import PageRankSimulator


class TestPageRankSimulator:
    """Test suite for PageRankSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator with default parameters."""
        return PageRankSimulator(num_nodes=5, edge_probability=0.5, seed=42)

    def test_generate_graph(self, simulator):
        """Test that graph is generated correctly."""
        graph = simulator.generate_graph()
        assert graph is not None
        assert graph.number_of_nodes() == 5
        assert graph.is_directed()

    def test_compute_adjacency_matrix(self, simulator):
        """Test adjacency matrix computation."""
        simulator.generate_graph()
        adj = simulator.compute_adjacency_matrix()
        assert adj.shape == (5, 5)
        assert np.all((adj >= 0) & (adj <= 1))

    def test_compute_transition_matrix(self, simulator):
        """Test transition matrix computation with damping factor."""
        simulator.generate_graph()
        simulator.compute_adjacency_matrix()
        trans = simulator.compute_transition_matrix(alpha=0.85)

        assert trans.shape == (5, 5)
        # Rows should sum to approximately 1 (stochastic)
        row_sums = trans.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)

    def test_compute_stationary_probability(self, simulator):
        """Test stationary probability computation."""
        simulator.generate_graph()
        simulator.compute_adjacency_matrix()
        simulator.compute_transition_matrix(alpha=0.85)

        stat = simulator.compute_stationary_probability(power=1000, alpha=0.85)
        assert len(stat) == 5
        # Probabilities should sum to approximately 1
        np.testing.assert_allclose(stat.sum(), 1.0, rtol=1e-5)
        assert all(0 <= p <= 1 for p in stat.values)

    def test_simulate_markov_chain(self, simulator):
        """Test Markov chain simulation."""
        simulator.generate_graph()
        simulator.compute_adjacency_matrix()
        simulator.compute_transition_matrix(alpha=0.85)

        chain = simulator.simulate_markov_chain(length=100)
        assert len(chain) == 101  # initial state + 100 steps
        assert all(state in simulator.social_networks[:5] for state in chain)

    def test_get_transition_string(self, simulator):
        """Test transition string format."""
        simulator.generate_graph()
        simulator.compute_adjacency_matrix()
        simulator.compute_transition_matrix(alpha=0.85)

        trans_str = simulator.get_transition_string(length=50)
        assert " -> " in trans_str
        parts = trans_str.split(" -> ")
        assert len(parts) == 51  # 50 transitions + 1 initial

    def test_different_seeds(self):
        """Test that different seeds produce different graphs."""
        sim1 = PageRankSimulator(num_nodes=5, edge_probability=0.5, seed=1)
        sim2 = PageRankSimulator(num_nodes=5, edge_probability=0.5, seed=2)

        sim1.generate_graph()
        sim2.generate_graph()

        adj1 = sim1.compute_adjacency_matrix()
        adj2 = sim2.compute_adjacency_matrix()

        # Different seeds should (very likely) produce different graphs
        assert not np.array_equal(adj1, adj2)

    def test_transition_matrix_alpha_effect(self):
        """Test that different alpha values produce different transition matrices."""
        sim1 = PageRankSimulator(num_nodes=5, edge_probability=0.5, seed=42)
        sim2 = PageRankSimulator(num_nodes=5, edge_probability=0.5, seed=42)

        sim1.generate_graph()
        sim1.compute_adjacency_matrix()
        sim2.generate_graph()
        sim2.compute_adjacency_matrix()

        trans1 = sim1.compute_transition_matrix(alpha=0.8)
        trans2 = sim2.compute_transition_matrix(alpha=0.9)

        assert not np.array_equal(trans1, trans2)

    def test_graph_deterministic_with_seed(self):
        """Test that same seed produces same graph."""
        sim1 = PageRankSimulator(num_nodes=5, edge_probability=0.5, seed=42)
        sim2 = PageRankSimulator(num_nodes=5, edge_probability=0.5, seed=42)

        sim1.generate_graph()
        sim2.generate_graph()

        adj1 = sim1.compute_adjacency_matrix()
        adj2 = sim2.compute_adjacency_matrix()

        np.testing.assert_array_equal(adj1, adj2)

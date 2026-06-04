"""Tests for the Markov model classes."""

import numpy as np
import pytest

from src.models.markov import (
    BaumWelch,
    HiddenMarkovChain,
    MarkovModel,
    Viterbi,
)


class TestMarkovModel:
    """Test suite for MarkovModel class."""

    @pytest.fixture
    def model(self):
        """Create a simple Markov model."""
        states = ["S1", "S2"]
        observations = ["O1", "O2", "O3"]
        transition = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
        return MarkovModel(states, observations, transition, emission)

    def test_observation_pairs(self, model):
        """Test that observation pairs are generated correctly."""
        pairs = model.observation_pairs
        assert len(pairs) > 0
        # Check no duplicates
        assert len(pairs) == len(set(pairs))
        # Check both orderings are excluded
        for i, j in pairs:
            assert (j, i) not in pairs or i == j

    def test_joint_emission_matrix_shape(self, model):
        """Test joint emission matrix dimensions."""
        assert model.joint_emission_matrix.shape[0] == model.transition_matrix.shape[0]

    def test_joint_emission_matrix_values(self, model):
        """Test that joint emission matrix has valid probability values."""
        assert np.all(model.joint_emission_matrix >= 0)
        assert np.all(model.joint_emission_matrix <= 1)


class TestHiddenMarkovChain:
    """Test suite for HiddenMarkovChain class."""

    @pytest.fixture
    def hmm(self):
        """Create a simple HMM."""
        transition = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
        return HiddenMarkovChain(
            sequence_length=50,
            transition_matrix=transition,
            emission_matrix=emission,
            num_simulations=3,
        )

    def test_simulate_output_shape(self, hmm):
        """Test that simulate returns correct shapes."""
        hidden, observed = hmm.simulate()
        assert hidden.shape == (50,)
        assert observed.shape == (50,)
        assert hidden.dtype == int
        assert observed.dtype == int

    def test_simulate_values_in_range(self, hmm):
        """Test that simulated values are within valid ranges."""
        hidden, observed = hmm.simulate()
        assert all(0 <= h < 2 for h in hidden)
        assert all(0 <= o < 3 for o in observed)

    def test_multiple_simulations_columns(self, hmm):
        """Test that multiple simulations produce expected number of columns."""
        df = hmm.simulate_multiple(
            state_names=["S1", "S2"],
            observation_pairs=[("O1", "O2"), ("O2", "O3"), ("O1", "O3")],
        )
        # 2 columns per simulation * 3 simulations = 6 columns
        assert df.shape[1] == 6


class TestBaumWelch:
    """Test suite for BaumWelch class."""

    @pytest.fixture
    def observations(self):
        """Create simple observation sequence."""
        return np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    @pytest.fixture
    def bw(self, observations):
        """Create a BaumWelch instance."""
        transition = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
        initial = np.array([0.5, 0.5])
        return BaumWelch(
            observations=observations,
            transition_matrix=transition,
            emission_matrix=emission,
            initial_distribution=initial,
            n_iterations=10,
        )

    def test_forward(self, bw):
        """Test forward algorithm output shape."""
        alpha = bw._forward()
        assert alpha.shape == (len(bw.observations), bw.transition_matrix.shape[0])
        assert np.all(alpha >= 0)

    def test_backward(self, bw):
        """Test backward algorithm output shape."""
        beta = bw._backward()
        assert beta.shape == (len(bw.observations), bw.transition_matrix.shape[0])
        assert np.all(beta > 0)

    def test_estimate_returns_matrices(self, bw):
        """Test that estimate returns both matrices."""
        result = bw.estimate()
        assert "transition_matrix" in result
        assert "em_matrix" in result
        assert result["transition_matrix"].shape == bw.transition_matrix.shape
        assert result["em_matrix"].shape == bw.emission_matrix.shape


class TestViterbi:
    """Test suite for Viterbi class."""

    def test_decode_returns_state_names(self):
        """Test that decode returns list of state names."""
        observations = np.array([0, 1, 2, 0, 1])
        transition = np.array([[0.7, 0.3], [0.4, 0.6]])
        emission = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
        initial = np.array([0.5, 0.5])
        state_names = ["S1", "S2"]

        decoded = Viterbi.decode(
            observations, transition, emission, initial, state_names
        )

        assert isinstance(decoded, list)
        assert len(decoded) == len(observations)
        assert all(name in state_names for name in decoded)

"""Hidden Markov Model module.

Contains classes for HMM simulation, Baum-Welch estimation,
and Viterbi decoding.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class MarkovModel:
    """Defines a Markov model with transition and emission matrices.

    Args:
        states: List of hidden state names
        observations: List of observable state names
        transition_matrix: State transition probabilities (n_states x n_states)
        emission_matrix: Observation emission probabilities (n_states x n_observations)
    """

    def __init__(
        self,
        states: List[str],
        observations: List[str],
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
    ) -> None:
        self.states = states
        self.observations = observations
        self.transition_matrix = transition_matrix
        # Ensure emission_matrix is (n_states x n_observations)
        n_states = len(states)
        n_obs = len(observations)
        if emission_matrix.shape == (n_states, n_obs):
            self.emission_matrix = emission_matrix.copy()
        elif emission_matrix.shape == (n_obs, n_states):
            self.emission_matrix = emission_matrix.T.copy()
        else:
            raise ValueError(
                f"Emission matrix shape {emission_matrix.shape} doesn't match "
                f"expected ({n_states}, {n_obs}) or ({n_obs}, {n_states})"
            )
        self.observation_pairs = self._generate_observation_pairs()
        self.joint_emission_matrix = self._generate_joint_emission_matrix()

    def _generate_observation_pairs(self) -> List[Tuple[str, str]]:
        """Generate unique pairs of observations (order-independent)."""
        pairs = []
        for obs_i in self.observations:
            for obs_j in self.observations:
                if (obs_i, obs_j) not in pairs and (obs_j, obs_i) not in pairs:
                    pairs.append((obs_i, obs_j))
        return pairs

    def _generate_joint_emission_matrix(self) -> np.ndarray:
        """Compute joint emission probabilities for observation pairs.

        Assumes independence between the two observation words
        given the hidden state.

        Returns:
            Joint emission matrix of shape (n_states, n_pairs)
        """
        n_states = self.emission_matrix.shape[0]
        n_obs = self.emission_matrix.shape[1]
        num_pairs = len(self.observation_pairs)
        joint_matrix = np.zeros((n_states, num_pairs))

        pair_index = 0
        for i in range(n_obs):
            for j in range(i, n_obs):
                if i != j:
                    joint_matrix[:, pair_index] = (
                        2.0 * self.emission_matrix[:, i] * self.emission_matrix[:, j]
                    )
                else:
                    joint_matrix[:, pair_index] = (
                        self.emission_matrix[:, i] * self.emission_matrix[:, j]
                    )
                pair_index += 1

        return joint_matrix


class HiddenMarkovChain:
    """Simulates a Hidden Markov Chain.

    Args:
        sequence_length: Number of steps in the chain
        transition_matrix: State transition probabilities
        emission_matrix: Observation emission probabilities
        num_simulations: Number of independent simulations to run
    """

    def __init__(
        self,
        sequence_length: int,
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        num_simulations: int = 1,
    ) -> None:
        self.sequence_length = sequence_length
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.num_simulations = num_simulations

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single HMM simulation.

        Returns:
            Tuple of (hidden_states, observations) arrays
        """
        num_states = self.emission_matrix.shape[0]
        num_obs = self.emission_matrix.shape[1]

        hidden_states = np.zeros(self.sequence_length, dtype=int)
        observations = np.zeros(self.sequence_length, dtype=int)

        # Initialize first state
        hidden_states[0] = np.random.choice(num_states)
        observations[0] = np.random.choice(
            num_obs,
            p=self.emission_matrix[hidden_states[0]],
        )

        # Generate remaining sequence
        for t in range(1, self.sequence_length):
            hidden_states[t] = np.random.choice(
                num_states,
                p=self.transition_matrix[hidden_states[t - 1]],
            )
            observations[t] = np.random.choice(
                num_obs,
                p=self.emission_matrix[hidden_states[t]],
            )

        return hidden_states, observations

    def simulate_multiple(
        self,
        state_names: List[str],
        observation_pairs: List[Tuple[str, str]],
    ) -> pd.DataFrame:
        """Generate multiple independent HMM simulations.

        Args:
            state_names: Names for the hidden states
            observation_pairs: Names for the observation pairs

        Returns:
            DataFrame with columns for each simulation's states and observations
        """
        dfs = []

        for sim_idx in range(self.num_simulations):
            hidden_states, observations = self.simulate()

            sim_states = [state_names[s] for s in hidden_states]
            sim_obs = [observation_pairs[o] for o in observations]

            df_sim = pd.DataFrame(
                {
                    f"States_{sim_idx}": sim_states,
                    f"Observations_{sim_idx}": sim_obs,
                }
            )
            dfs.append(df_sim)

        return pd.concat(dfs, axis=1)


class BaumWelch:
    """Implements the Baum-Welch algorithm for HMM parameter estimation.

    Args:
        observations: Observed sequence (1D array of integer indices)
        transition_matrix: Initial transition matrix estimate
        emission_matrix: Initial emission matrix estimate
        initial_distribution: Initial state distribution
        n_iterations: Maximum number of EM iterations
    """

    def __init__(
        self,
        observations: np.ndarray,
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        initial_distribution: np.ndarray,
        n_iterations: int = 100,
    ) -> None:
        self.observations = observations
        self.transition_matrix = transition_matrix.copy()
        self.emission_matrix = emission_matrix.copy()
        self.initial_distribution = initial_distribution.copy()
        self.n_iterations = n_iterations

    def _forward(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scaled forward probabilities (alpha).

        Returns:
            Tuple of (scaled_alpha, scaling_factors)
        """
        T = len(self.observations)
        M = self.transition_matrix.shape[0]
        alpha = np.zeros((T, M))
        c = np.zeros(T)

        # Initialization
        alpha[0] = (
            self.initial_distribution * self.emission_matrix[:, self.observations[0]]
        )
        c[0] = 1.0 / (np.sum(alpha[0]) + 1e-15)
        alpha[0] *= c[0]

        # Induction
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.transition_matrix) * self.emission_matrix[
                :, self.observations[t]
            ]
            c[t] = 1.0 / (np.sum(alpha[t]) + 1e-15)
            alpha[t] *= c[t]

        return alpha, c

    def _backward(self, c: np.ndarray) -> np.ndarray:
        """Compute scaled backward probabilities (beta).

        Args:
            c: Scaling factors from the forward pass

        Returns:
            Scaled backward probability matrix (T x M)
        """
        T = len(self.observations)
        M = self.transition_matrix.shape[0]
        beta = np.zeros((T, M))

        # Initialization
        beta[T - 1] = c[T - 1]

        # Induction (backward in time)
        for t in range(T - 2, -1, -1):
            beta[t] = (
                self.transition_matrix
                @ (self.emission_matrix[:, self.observations[t + 1]] * beta[t + 1])
            ) * c[t]

        return beta

    def estimate(self) -> Dict[str, np.ndarray]:
        """Run the Baum-Welch algorithm to estimate HMM parameters.

        Returns:
            Dictionary with estimated 'transition_matrix' and 'emission_matrix'
        """
        M = self.transition_matrix.shape[0]
        T = len(self.observations)
        K = self.emission_matrix.shape[1]

        for _ in range(self.n_iterations):
            alpha, c = self._forward()
            beta = self._backward(c)

            # Fully vectorized computation of xi (joint state probabilities)
            # xi shape: (T-1, M, M)
            # xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, o_{t+1}] * beta[t+1, j]
            obs_next = self.observations[1:]  # shape (T-1,)
            # alpha[:-1]: (T-1, M) → (T-1, M, 1)
            # self.transition_matrix: (M, M)
            # self.emission_matrix[:, obs_next]: (M, T-1) → (T-1, M) for broadcast as (T-1, 1, M)
            # beta[1:]: (T-1, M) → (T-1, 1, M)
            num_xi = (
                alpha[:-1, :, None]
                * self.transition_matrix[None, :, :]
                * self.emission_matrix[:, obs_next].T[:, None, :]
                * beta[1:, None, :]
            )  # shape (T-1, M, M)
            denom_xi = num_xi.sum(axis=(1, 2), keepdims=True) + 1e-15
            xi = num_xi / denom_xi

            # Compute gamma (marginal state probabilities)
            # gamma shape: (T, M)
            gamma = np.zeros((T, M))
            gamma[: T - 1] = np.sum(xi, axis=2)
            # Last gamma: normalize the already-scaled alpha[T-1]
            gamma[T - 1] = alpha[T - 1] / (np.sum(alpha[T - 1]) + 1e-15)

            # Update transition matrix A
            # A[i, j] = sum_t(xi[t, i, j]) / sum_t(gamma[t, i])
            num_a = np.sum(xi, axis=0)
            den_a = np.sum(gamma[: T - 1], axis=0)[:, None]
            self.transition_matrix = num_a / (den_a + 1e-15)

            # Update emission matrix B
            # B[i, k] = sum_{t: o_t=k}(gamma[t, i]) / sum_t(gamma[t, i])
            num_b = np.zeros((M, K))
            for k in range(K):
                num_b[:, k] = np.sum(gamma[self.observations == k], axis=0)

            den_b = np.sum(gamma, axis=0)[:, None]
            self.emission_matrix = num_b / (den_b + 1e-15)

        return {
            "a": np.around(self.transition_matrix, 4),
            "b": np.around(self.emission_matrix, 4),
        }


class Viterbi:
    """Implements the Viterbi algorithm for HMM decoding.

    Finds the most likely sequence of hidden states given
    an observation sequence.
    """

    @staticmethod
    def decode(
        observations: np.ndarray,
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        initial_distribution: np.ndarray,
        state_names: List[str],
    ) -> List[str]:
        """Decode the most likely hidden state sequence.

        Args:
            observations: Observed sequence (1D array)
            transition_matrix: State transition probabilities
            emission_matrix: Observation emission probabilities
            initial_distribution: Initial state distribution
            state_names: Names for the hidden states

        Returns:
            List of decoded state names
        """
        T = len(observations)
        M = transition_matrix.shape[0]

        # Initialize omega and backpointer matrices
        omega = np.zeros((T, M))
        prev = np.zeros((T, M), dtype=int)

        # Initialization step
        omega[0] = np.log(initial_distribution * emission_matrix[:, observations[0]])

        # Recursion step
        for t in range(1, T):
            for j in range(M):
                probabilities = (
                    omega[t - 1]
                    + np.log(transition_matrix[:, j])
                    + np.log(emission_matrix[j, observations[t]])
                )
                prev[t, j] = np.argmax(probabilities)
                omega[t, j] = np.max(probabilities)

        # Termination step
        state_sequence = np.zeros(T, dtype=int)
        state_sequence[T - 1] = np.argmax(omega[T - 1])

        # Backtracking step
        for t in range(T - 2, -1, -1):
            state_sequence[t] = prev[t + 1, state_sequence[t + 1]]

        # Convert to state names
        return [state_names[s] for s in state_sequence]


class WebCommunitySimulator:
    """Simulates web communities using Stochastic Block Models (SBM).

    Args:
        n_nodes: Number of nodes/pages in the community
        pi: Community proportions (should sum to 1)
        alpha: Within-community edge probability
        beta: Between-community edge probability
    """

    def __init__(
        self,
        n_nodes: int = 90,
        pi: List[float] = None,
        alpha: float = 0.15,
        beta: float = 0.05,
    ) -> None:
        self.n_nodes = n_nodes
        self.pi = pi or [1 / 3, 1 / 3, 1 / 3]
        self.alpha = alpha
        self.beta = beta
        self.hidden_states: np.ndarray | None = None
        self.adjacency_matrix: np.ndarray | None = None

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a community simulation using SBM.

        Returns:
            Tuple of (hidden_states, adjacency_matrix)
        """
        # Sample community assignments
        community_counts = np.random.multinomial(self.n_nodes, self.pi)
        hidden_states = np.concatenate(
            [
                np.full(count, community_id + 1)
                for community_id, count in enumerate(community_counts)
            ]
        )

        # Generate adjacency matrix (vectorized)
        # Build probability matrix: alpha on diagonal blocks, beta elsewhere
        same_community = (hidden_states[:, None] == hidden_states[None, :])
        prob_matrix = np.where(same_community, self.alpha, self.beta)
        np.fill_diagonal(prob_matrix, 0.0)  # no self-loops
        adjacency = np.random.binomial(1, prob_matrix).astype(float)

        self.hidden_states = hidden_states
        self.adjacency_matrix = adjacency
        return hidden_states, adjacency

    def generate_observations(
        self,
        observation_pairs: List[Tuple[str, str]],
        emission_matrix: np.ndarray,
    ) -> List[str]:
        """Generate observable words from hidden community states.

        Args:
            observation_pairs: Pairs of possible observation words
            emission_matrix: Joint emission probabilities

        Returns:
            List of observed word strings
        """
        if self.hidden_states is None:
            raise ValueError("Call simulate() first.")

        observations = []
        n_emission_categories = emission_matrix.shape[1]

        for state in self.hidden_states:
            obs_idx = np.random.choice(
                n_emission_categories,
                p=emission_matrix[int(state) - 1],
            )
            # Return the pair as a string representation
            pair = observation_pairs[obs_idx]
            observations.append(f"{pair[0]}-{pair[1]}")

        return observations

    def compute_transition_matrices(
        self,
        inv_epsilon: float = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute two transition matrices for the community simulation.

        A1: Normalized adjacency with additive smoothing (Laplace-like)
        A2: Uniform transition matrix

        Args:
            inv_epsilon: Smoothing intensity (higher = more smoothing; equivalent
                to the former ``1 / epsilon`` parameterization). Default: 1000.

        Returns:
            Tuple of (A1, A2) matrices
        """
        if self.adjacency_matrix is None:
            raise ValueError("Call simulate() first.")

        eps = inv_epsilon
        A1 = (self.adjacency_matrix + eps) / (
            np.sum(self.adjacency_matrix, axis=1) + self.n_nodes * eps
        )[:, None]
        A1 = np.nan_to_num(A1)
        A2 = np.ones((self.n_nodes, self.n_nodes)) / self.n_nodes

        return A1, A2

    def simulate_random_walk(
        self,
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        observation_pairs: List[Tuple[str, str]],
        n_steps: int = 500,
    ) -> Tuple[List[str], List[float]]:
        """Simulate a random walk on the community graph.

        Args:
            transition_matrix: Transition probabilities (n_nodes x n_nodes)
            emission_matrix: Joint emission probabilities (n_communities x n_pairs)
            observation_pairs: Available observation pairs
            n_steps: Number of walk steps

        Returns:
            Tuple of (walk_observations, normalized_positions)
        """
        if self.hidden_states is None:
            raise ValueError("Call simulate() first.")

        n_states = transition_matrix.shape[0]
        n_communities = emission_matrix.shape[0]
        current_state = np.random.choice(n_states)

        def _community(node: int) -> int:
            """Map node index to 0-based community index."""
            return int(self.hidden_states[node]) - 1

        # Sample first observation based on the community of the current node
        community = _community(current_state)
        obs_idx = np.random.choice(n_communities, p=emission_matrix[community])
        pair = observation_pairs[obs_idx]

        observations = [f"{pair[0]}-{pair[1]}"]
        positions = [current_state / max(n_states - 1, 1)]

        for _ in range(n_steps - 1):
            current_state = np.random.choice(
                n_states,
                p=transition_matrix[current_state],
            )
            community = _community(current_state)
            obs_idx = np.random.choice(n_communities, p=emission_matrix[community])
            pair = observation_pairs[obs_idx]
            observations.append(f"{pair[0]}-{pair[1]}")
            positions.append(current_state / max(n_states - 1, 1))

        return observations, positions

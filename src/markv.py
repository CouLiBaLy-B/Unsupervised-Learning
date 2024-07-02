import numpy as np
import pandas as pd
import streamlit as st


class MarkovModel:
    def __init__(self, domaines, mots, A, B):
        self.domaines = domaines
        self.mots = mots
        self.A = A
        self.B = B
        self.keyword_pairs = self._generate_keyword_pairs()
        self.D = self._generate_emission_matrix()

    def _generate_keyword_pairs(self):
        keyword_pairs = []
        for i in self.mots:
            for j in self.mots:
                if (i, j) not in keyword_pairs and (j, i) not in keyword_pairs:
                    keyword_pairs.append((i, j))
        return keyword_pairs

    def _generate_emission_matrix(self):
        somme = np.sum(range(len(self.mots) + 1))
        D = np.zeros((self.A.shape[0], somme))
        for i in range(self.B.shape[0]):
            for j in range(self.B.shape[1]):
                for k in range(j, self.B.shape[1]):
                    if j != k:
                        D[i][np.sum(range(self.B.shape[1])) - np.sum(range(self.B.shape[1] - j)) + k] = 2 * self.B[i][j] * self.B[i][k]
                    else:
                        D[i][np.sum(range(self.B.shape[1])) - np.sum(range(self.B.shape[1] - j)) + k] = self.B[i][j] * self.B[i][k]
        return D


class HiddenMarkovChain:
    def __init__(self, n, A, B, m):
        self.n = n
        self.A = A
        self.B = B
        self.m = m

    def simulate(self):
        M = self.B.shape[0]
        k = self.B.shape[1]
        Z = np.zeros(self.n, dtype=int)
        X = np.zeros(self.n, dtype=int)
        Z[0] = np.random.choice(range(M))
        X[0] = np.random.choice(range(k), p=self.B[Z[0],])
        for i in range(1, self.n):
            Z[i] = np.random.choice(range(M), 1, p=self.A[Z[i-1],])
            X[i] = np.random.choice(range(k), p=self.B[Z[i],])
        return np.array([Z, X])

    def multiple_simulations(self, domaines, keyword_pairs):
        df = pd.DataFrame(self.simulate().T, columns=["Domaines_0", "Mots clefs_0"])
        df["Domaines_0"] = df["Domaines_0"].apply(lambda x: domaines[x])
        df["Mots clefs_0"] = df["Mots clefs_0"].apply(lambda x: keyword_pairs[x])
        for i in range(1, self.m):
            df[f"Domaines_{i}"], df[f"Mots clefs_{i}"] = self.simulate()
            df[f"Domaines_{i}"] = df[f"Domaines_{i}"].apply(lambda x: domaines[x])
            df[f"Mots clefs_{i}"] = df[f"Mots clefs_{i}"].apply(lambda x: keyword_pairs[x])
        return df


class BaumWelch:
    def __init__(self, V, a, b, initial, n_iter):
        self.V = V
        self.a = a
        self.b = b
        self.initial = initial
        self.n_iter = n_iter

    def forward(self):
        alpha = np.zeros((self.V.shape[0], self.a.shape[0]))
        alpha[0, :] = self.initial* self.b[:, self.V[0]]
        for t in range(1, self.V.shape[0]):
            for j in range(self.a.shape[0]):
                alpha[t, j] = alpha[t - 1].dot(self.a[:, j]) * self.b[j, self.V[t]]
        return alpha

    def backward(self):
        beta = np.ones((self.V.shape[0], self.a.shape[0]))
        beta[self.V.shape[0] - 1] = np.ones((self.a.shape[0]))
        for t in range(self.V.shape[0] - 2, -1, -1):
            for j in range(self.a.shape[0]):
                beta[t, j] = (beta[t + 1] * self.b[:, self.V[t + 1]]).dot(self.a[j, :])
        return beta

    def baum_welch(self):
        M = self.a.shape[0]
        T = len(self.V)
        for n in range(self.n_iter):
            alpha = BaumWelch.forward(self)
            beta = BaumWelch.backward(self)
            xi = np.zeros((M, M, T - 1))
            for t in range(T - 1):
                denominator = alpha[t, :].T @ self.a * self.b[:, self.V[t + 1]].T @ beta[t + 1, :]
                for i in range(M):
                    numerator = alpha[t, i] * self.a[i, :] * self.b[:, self.V[t + 1]].T * beta[t + 1, :].T
                    xi[i, :, t] = numerator / denominator
            gamma = np.sum(xi, axis=1)
            self.a = np.around(np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1)),4)
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
            K = self.b.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                self.b[:, l] = np.sum(gamma[:, self.V == l], axis=1)
            self.b = np.around(np.divide(self.b, denominator.reshape((-1, 1))), 4)
        return {"a": self.a, "b": self.b}


class Viterbi:
    @staticmethod
    def decode(V, a, b, initial_distribution, domaines):
        T = V.shape[0]
        M = a.shape[0]
        omega = np.zeros((T, M))
        omega[0, :] = np.log(initial_distribution * b[:, V[0]])
        prev = np.zeros((T - 1, M))
        for t in range(1, T):
            for j in range(M):
                probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
                prev[t - 1, j] = np.argmax(probability)
                omega[t, j] = np.max(probability)
        S = np.zeros(T)
        last_state = np.argmax(omega[T - 1, :])
        S[0] = last_state
        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            S[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1
        S = np.flip(S, axis=0)
        result = []
        for s in S:
            result.append(domaines[int(s)])
        return result

class WebSimulation:
    def __init__(self, markov_model):
        self.markov_model = markov_model

    def run_simulation(self, n, m):
        hmm = HiddenMarkovChain(n, self.markov_model.A, self.markov_model.D, m)
        simulation_results = hmm.multiple_simulations(self.markov_model.domaines, self.markov_model.keyword_pairs)
        return simulation_results

    def estimate_parameters(self, observations):
        initial_distribution = np.ones(self.markov_model.A.shape[0]) / self.markov_model.A.shape[0]
        bw = BaumWelch(observations, self.markov_model.A, self.markov_model.D, initial_distribution, n_iter=100)
        estimated_params = bw.baum_welch()
        return estimated_params

    def decode_states(self, observations):
        initial_distribution = np.ones(self.markov_model.A.shape[0]) / self.markov_model.A.shape[0]
        decoded_states = Viterbi.decode(observations, self.markov_model.A, self.markov_model.D, initial_distribution, self.markov_model.domaines)
        return decoded_states

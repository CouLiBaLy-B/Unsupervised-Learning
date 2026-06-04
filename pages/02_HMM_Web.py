"""Streamlit page - Web Communication with Hidden Markov Model."""

import numpy as np
import pandas as pd
import streamlit as st

from src.models.markov import (
    BaumWelch,
    HiddenMarkovChain,
    MarkovModel,
    Viterbi,
)
from src.utils.config import (
    DEFAULT_DOMAINS,
    DEFAULT_EMISSION_MATRIX,
    DEFAULT_HMM_SIMULATION_LENGTH,
    DEFAULT_HMM_SIMULATION_WIDTH,
    DEFAULT_KEYWORDS,
    DEFAULT_TRANSITION_MATRIX,
)

st.set_page_config(
    page_title="HMM Web Simulation - Unsupervised Learning",
    page_icon="🌐",
    layout="wide",
)

st.markdown(
    "<h1 style='text-align: center; background-color: #2D3E50; color: #FFFFFF'>"
    "2. Communication Web - Chaîne de Markov Cachée</h1>",
    unsafe_allow_html=True,
)


def main() -> None:
    """Run the HMM web communication simulation app."""
    st.write(
        """Dans cette section, nous simulons un parcours web à l'aide d'une chaîne
        de Markov cachée (HMM) dont les paramètres sont les états cachés (domaines)
        et les observations (mots clés)."""
    )

    # ---- Sidebar Parameters ----
    st.sidebar.header("Paramètres HMM")
    n_sim = st.sidebar.number_input(
        "Taille de la simulation (longueur)",
        min_value=10,
        max_value=200,
        value=DEFAULT_HMM_SIMULATION_LENGTH,
    )
    m_sim = st.sidebar.number_input(
        "Largeur de la simulation (nombre de parcours)",
        min_value=1,
        max_value=100,
        value=DEFAULT_HMM_SIMULATION_WIDTH,
    )
    n_iter_bw = st.sidebar.number_input(
        "Nombre d'itérations Baum-Welch",
        min_value=10,
        max_value=500,
        value=100,
    )

    # ---- Define Model ----
    domaines = DEFAULT_DOMAINS
    mots = DEFAULT_KEYWORDS
    A = np.array(DEFAULT_TRANSITION_MATRIX, dtype=float)
    # Emission matrix: (n_states, n_observations) = (3, 10)
    B = np.array(DEFAULT_EMISSION_MATRIX, dtype=float)

    st.subheader("États cachés (Domaines)")
    st.dataframe(pd.DataFrame(domaines, columns=["Domaine"]))

    st.subheader("Matrice de transition A")
    st.dataframe(pd.DataFrame(A, index=domaines, columns=domaines))

    st.subheader("États visibles (Mots clés)")
    st.dataframe(pd.DataFrame(mots, columns=["Mot clé"]))

    st.subheader("Matrice d'émission B")
    st.dataframe(pd.DataFrame(B, index=domaines, columns=mots))

    # ---- Initialize Markov Model ----
    markov_model = MarkovModel(
        states=domaines,
        observations=mots,
        transition_matrix=A,
        emission_matrix=B,
    )

    # ---- Joint Emission Matrix ----
    st.subheader("Matrice de probabilité jointe D")
    st.write(
        """Pour la simulation à double mots clés, nous déterminons la matrice
        des probabilités pour des couples de mots clés, sous condition d'indépendance
        des deux mots clés par rapport au domaine :

        $D_{i,j} = P(X_j = W1, Y_j = W2 | Dom_i) = P(X_j = W1 | Dom_i) \\times P(Y_j = W2 | Dom_i)$
        """
    )

    keyword_pairs = markov_model.observation_pairs
    joint_matrix = markov_model.joint_emission_matrix

    st.dataframe(
        pd.DataFrame(
            joint_matrix,
            index=domaines,
            columns=[f"{p[0]}-{p[1]}" for p in keyword_pairs],
        )
    )

    st.write("Vérification du caractère stochastique de la matrice D:")
    st.write(f"Sommes par ligne: {joint_matrix.sum(axis=1)}")

    # ---- Run Simulation ----
    st.subheader("Simulation de la chaîne de Markov cachée")

    if st.button("Lancer la simulation"):
        hmm = HiddenMarkovChain(
            sequence_length=n_sim,
            transition_matrix=markov_model.transition_matrix,
            emission_matrix=joint_matrix,
            num_simulations=m_sim,
        )

        with st.spinner("Simulation en cours..."):
            results = hmm.simulate_multiple(domaines, keyword_pairs)
            st.dataframe(results)

        # ---- Baum-Welch Estimation ----
        st.subheader("Estimation des paramètres avec Baum-Welch")

        # Recode results into integer indices
        mat_encoded = results.copy()
        for i in range(mat_encoded.shape[0]):
            for j in range(0, mat_encoded.shape[1], 2):
                state_val = mat_encoded.iloc[i, j]
                obs_val = mat_encoded.iloc[i, j + 1]

                if isinstance(state_val, str):
                    mat_encoded.iloc[i, j] = domaines.index(state_val)
                else:
                    mat_encoded.iloc[i, j] = int(state_val)

                if isinstance(obs_val, str):
                    mat_encoded.iloc[i, j + 1] = keyword_pairs.index(obs_val)
                else:
                    mat_encoded.iloc[i, j + 1] = int(obs_val)

        if st.checkbox("Afficher la matrice recodée", value=False):
            st.dataframe(mat_encoded)

        # Run Baum-Welch on first observation column
        V = mat_encoded.iloc[:, 1].values.astype(int)
        initial_dist = np.ones(A.shape[0]) / A.shape[0]

        with st.spinner("Baum-Welch en cours..."):
            bw = BaumWelch(
                observations=V,
                transition_matrix=A.copy(),
                emission_matrix=joint_matrix.copy(),
                initial_distribution=initial_dist,
                n_iterations=n_iter_bw,
            )
            estimated = bw.estimate()

        st.write("Les estimateurs de Baum-Welch de A et D sont:")
        st.write("Matrice A estimée:")
        st.dataframe(pd.DataFrame(estimated["a"], index=domaines, columns=domaines))
        st.write("Matrice D estimée:")
        st.dataframe(pd.DataFrame(estimated["b"]))

        # ---- Viterbi Decoding ----
        st.subheader("Générateur des états cachés avec Viterbi")
        st.write(
            """L'algorithme de Viterbi est une solution optimale au sens du maximum
            de vraisemblance pour l'estimation d'une séquence d'états d'un processus
            de Markov à temps discret et nombre d'états finis."""
        )

        with st.spinner("Viterbi en cours..."):
            decoded = Viterbi.decode(
                observations=V,
                transition_matrix=estimated["a"],
                emission_matrix=estimated["b"],
                initial_distribution=initial_dist,
                state_names=domaines,
            )
            st.dataframe(pd.DataFrame({"États décodés": decoded}))


if __name__ == "__main__":
    main()

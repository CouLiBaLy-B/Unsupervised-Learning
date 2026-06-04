"""Streamlit page - SBM Web Community Simulation."""

import numpy as np
import pandas as pd
import streamlit as st

from src.models.markov import MarkovModel, WebCommunitySimulator
from src.utils.config import (
    DEFAULT_DOMAINS,
    DEFAULT_EMISSION_MATRIX,
    DEFAULT_KEYWORDS,
    DEFAULT_SBM_ALPHA,
    DEFAULT_SBM_BETA,
    DEFAULT_SBM_SIZE,
)

st.set_page_config(
    page_title="SBM Community Simulation",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    "<h1 style='text-align: center; background-color: #2D3E50; color: #FFFFFF'>"
    "3. Simulation des Communautés Web - Stochastic Block Model</h1>",
    unsafe_allow_html=True,
)


def main() -> None:
    """Run the SBM community simulation app."""
    st.write("""Supposons qu'un ensemble de pages web soit partagé en groupes selon un
        processus stochastique :

        - Les pages traitant de sport (z = S)
        - Les pages traitant de culture (z = C)
        - Les pages traitant de soins beauté (z = B)

        Avec $X_{ij} | z_i=k, z_j=l \\sim B(\\alpha I_{(k=l)} + \\beta I_{(k \\neq l)})$
        et $P(z_i = k) = \\pi_k = 1/3$.
        """)

    # ---- Sidebar Parameters ----
    st.sidebar.header("Paramètres SBM")
    n_nodes = st.sidebar.number_input(
        "Taille n de la simulation",
        min_value=10,
        max_value=1000,
        value=DEFAULT_SBM_SIZE,
    )
    alpha = st.sidebar.slider(
        "Alpha (intra-communauté)",
        0.01,
        0.50,
        DEFAULT_SBM_ALPHA,
        key="alpha_sbm_slider",
    )
    beta = st.sidebar.slider(
        "Beta (inter-communauté)", 0.01, 0.50, DEFAULT_SBM_BETA, key="beta_sbm_slider"
    )
    epsilon_val = st.sidebar.number_input(
        "Epsilon (lissage = 1/n)",
        min_value=100,
        max_value=10000,
        value=1000,
    )
    walk_steps = st.sidebar.number_input(
        "Nombre de pas de marche aléatoire",
        min_value=100,
        max_value=10000,
        value=500,
    )

    # ---- Initialize Model ----
    markov_model = MarkovModel(
        states=DEFAULT_DOMAINS,
        observations=DEFAULT_KEYWORDS,
        transition_matrix=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.25, 0.7, 0.05],
                [0.1, 0.1, 0.8],
            ]
        ),
        emission_matrix=np.array(DEFAULT_EMISSION_MATRIX),
    )

    simulator = WebCommunitySimulator(
        n_nodes=n_nodes,
        alpha=alpha,
        beta=beta,
    )

    # ---- Run Simulation ----
    if st.button("Simuler la communauté"):
        with st.spinner("Simulation en cours..."):
            hidden_states, adj_matrix = simulator.simulate()
            st.subheader("Domaines (états cachés)")
            st.dataframe(pd.DataFrame({"Domaine": hidden_states}))
            st.subheader("Matrice d'adjacence X")
            st.dataframe(pd.DataFrame(adj_matrix))

            # Generate observations
            observations = simulator.generate_observations(
                observation_pairs=markov_model.observation_pairs,
                emission_matrix=markov_model.joint_emission_matrix,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.write("Domaines (mots cachés)")
                st.dataframe(pd.DataFrame({"Domaine": hidden_states}))
            with col2:
                st.write("Mots visibles")
                st.dataframe(pd.DataFrame({"Mot": observations}))

            # Transition matrices
            A1, A2 = simulator.compute_transition_matrices(epsilon_val)
            st.subheader("Matrices de transition A1 et A2")
            st.write(f"A1 (lissage ϵ={1/epsilon_val}):")
            st.dataframe(pd.DataFrame(A1))

            st.write("A2 (uniforme):")
            st.dataframe(pd.DataFrame(A2))

            # Random walks
            st.subheader("Simulations de marche aléatoire")
            walk_A1, pos_A1 = simulator.simulate_random_walk(
                transition_matrix=A1,
                emission_matrix=markov_model.joint_emission_matrix,
                observation_pairs=markov_model.observation_pairs,
                n_steps=walk_steps,
            )
            walk_A2, pos_A2 = simulator.simulate_random_walk(
                transition_matrix=A2,
                emission_matrix=markov_model.joint_emission_matrix,
                observation_pairs=markov_model.observation_pairs,
                n_steps=walk_steps,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Simulation avec A1 ({walk_steps} pas)")
                st.dataframe(pd.DataFrame({"Observation": walk_A1}))
            with col2:
                st.write(f"Simulation avec A2 ({walk_steps} pas)")
                st.dataframe(pd.DataFrame({"Observation": walk_A2}))


if __name__ == "__main__":
    main()

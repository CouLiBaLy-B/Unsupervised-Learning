import streamlit as st
import numpy as np
from src.markv import MarkovModel, WebSimulation


# Usage
domaine = ["Sport", "Culture", "Beauté"]
Mots = ["Abdominaux", "Cosmétiques", "Livres",
        "Age", "Force", "Endurance", "Résilience",
        "Crème", "Histoire", "Mathématiques"]
A = np.array([0.7, 0.2, 0.1, 0.25, 0.7,
              0.05, 0.1, 0.1, 0.8], dtype=float).reshape((3, 3))
B = np.array([0.2, 0, 0.1, 0, 0.1, 0.3, 0.1,
              0.3, 0, 0.1, 0.2, 0.2, 0.2, 0,
              0.1, 0.3, 0, 0, 0.1, 0.1, 0.1,
              0, 0, 0.2, 0, 0.2, 0, 0, 0.1, 0], 
              dtype=float).reshape((10, 3)
                                   ).T

markov_model = MarkovModel(domaine, Mots, A, B)
web_simulation = WebSimulation(markov_model)

# Streamlit app
st.markdown(
    """<h1 style='text-align: center; background-color: #2D3E50; color: #FFFFFF'>
    Web Simulation avec Chaîne de Markov Cachée
    </h1>""",
    unsafe_allow_html=True,
)
n = st.number_input(
    "Taille de la simulation",
    min_value=10,
    max_value=200,
    value=100
    )
m = st.number_input(
    "Largeur de la simulation",
    min_value=1,
    max_value=100,
    value=30
    )

if st.button("Lancer la simulation"):
    results = web_simulation.run_simulation(n, m)
    st.dataframe(results)

    # Recodage des résultats
    mat = results.copy()
    for i in range(mat.shape[0]):
        for j in range(0, mat.shape[1], 2):
            mat.iloc[i, j] = markov_model.domaines.index(results.iloc[i, j])
            mat.iloc[i, j+1] = markov_model.keyword_pairs.index(results.iloc[i, j+1])

    # Estimation des paramètres
    V = mat["Mots clefs_0"].values
    estimated_params = web_simulation.estimate_parameters(V)
    st.write(
        "Estimateurs de Baum-Welch:"
        )
    st.write(
        "Matrice A estimée:",
        estimated_params["a"]
        )
    st.write(
        "Matrice B estimée:",
        estimated_params["b"]
        )

    # Décodage des états
    decoded_states = web_simulation.decode_states(V)
    st.write("États décodés avec Viterbi:")
    st.dataframe(decoded_states)
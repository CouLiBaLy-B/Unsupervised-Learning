"""Streamlit Home page - Main entry point."""

import os

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Unsupervised Learning",
    page_icon="😎",
    layout="wide",
)

st.markdown(
    "<h1 style='text-align: center; background-color: #2D3E50; color: #FFFFFF'>"
    "Projet d'Unsupervised Learning</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='text-align: center; background-color: #2D3E50; color: #FFFFFF'>"
    "Ibrahim COULIBALY<br>M2 Data Science<br>Université Paris Saclay</h2>",
    unsafe_allow_html=True,
)

# Load image if it exists
image_path = os.path.join(os.path.dirname(__file__), "..", "unsupervised.jpeg")
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image)

st.write(
    """Cette application web a été créée avec la librairie **Streamlit** dans l'objectif
    de présenter et d'utiliser les résultats du projet Unsupervised Learning.
    """
)

st.write(
    """L'objectif de ce projet est d'appliquer les différents concepts vus durant le cours
    d'Unsupervised Learning, qui sont entre autres :

- **Chaîne de Markov** et **Chaîne de Markov cachée** pour les simulations de communauté
- **Algorithme de Baum-Welch** pour l'estimation et **Viterbi** pour une simulation
- **SBM (Stochastic Block Model)** pour la simulation également de communauté web
- **Deep Learning** : Perceptron Multicouche (MLP) et RNN pour la classification

Pour ce qui concerne la rédaction, nous allons suivre la même trame que les questions
dans le sujet du projet et apporter un peu de généralisation sur les choix des
différents paramètres fixés dans le sujet.
"""
)

st.header("📌 Navigation")

st.markdown("""
| Page | Description |
|------|-------------|
| 🔗 PageRank | Simulation de l'algorithme PageRank sur un graphe aléatoire |
| 🌐 HMM Web | Communication Web avec chaîne de Markov cachée |
| 🌍 SBM Community | Simulation des communautés web avec Stochastic Block Model |
| 🧠 Classifier | Classification avec Perceptron Multicouche (MLP) |
""")

st.write("---")
st.write("*Fin — Merci pour votre lecture !*")
st.write(
    "📎 Lien pour le code source du projet : "
    "[GitHub Repository](https://github.com/CouLiBaLy-B/Unsupervised-Learning)"
)

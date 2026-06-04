"""Streamlit page - PageRank simulation."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.simulations.pagerank import PageRankSimulator

st.set_page_config(
    page_title="PageRank - Unsupervised Learning",
    page_icon="🔗",
    layout="wide",
)

st.markdown(
    "<h1 style='text-align: center; background-color: #2D3E50; color: #FFFFFF'>"
    "1. Algorithme PageRank</h1>",
    unsafe_allow_html=True,
)

st.write("""Simulation d'un graphe orienté de N sommets et de probabilité p.
    Pour cette simulation, nous utilisons la librairie `networkx` de Python,
    équivalent du package R `igraph`.""")

# ---- Sidebar Parameters ----
st.sidebar.header("Paramètres")
num_nodes = st.sidebar.slider("Nombre de sommets", 7, 10, 8)
edge_prob = st.sidebar.slider("Probabilité p", 0.4, 1.0, 0.5)
alpha = st.sidebar.slider("Facteur d'amortissement α", 0.0, 1.0, 0.85)
power_iter = st.sidebar.number_input(
    "Puissance n de la matrice de transition",
    min_value=10,
    max_value=10000,
    value=1000,
)
chain_length = st.sidebar.number_input(
    "Taille de la simulation de la chaîne",
    min_value=10,
    max_value=10000,
    value=1000,
)

# ---- Initialize Simulator ----
simulator = PageRankSimulator(
    num_nodes=num_nodes,
    edge_probability=edge_prob,
    seed=20222023,
)

# ---- Generate Graph ----
st.header("Graphe simulé")
graph = simulator.generate_graph()

fig_graph, ax_graph = plt.subplots(figsize=(6, 6))
pos_layout = nx.spring_layout(graph, seed=42)
nx.draw(
    graph,
    pos=pos_layout,
    ax=ax_graph,
    node_size=800,
    node_color="blue",
    edge_color="red",
    with_labels=True,
    font_color="white",
    font_size=8,
)
st.pyplot(fig_graph)
plt.close(fig_graph)

# ---- Transition Matrix ----
st.subheader("La matrice de transition pour un ϵ donné")
st.write("""Une fois le graphe généré, nous pouvons facilement obtenir la matrice
    d'adjacence via `networkx.adjacency_matrix`, puis la transformer en matrice
    de transition avec lissage ϵ.""")

simulator.compute_adjacency_matrix()
transition_matrix = simulator.compute_transition_matrix(alpha)
st.dataframe(transition_matrix)

# ---- Stationary Probability ----
st.subheader("La probabilité stationnaire")
st.write(
    """Notre méthode de calcul de la probabilité stationnaire utilise les puissances
    de la matrice de transition. Par la définition de la probabilité stationnaire
    π = πA et par récurrence π = πA^n pour n assez grand, nous obtenons un résultat
    précis de la probabilité stationnaire."""
)

stationary_prob = simulator.compute_stationary_probability(
    power=power_iter,
    alpha=alpha,
)
st.dataframe(stationary_prob)

# ---- Markov Chain Simulation ----
st.subheader("Simulation d'une chaîne de Markov")
st.write(
    "Pour mettre dans un exemple concret, nous utilisons les réseaux sociaux les plus connus :"
)

networks = simulator.social_networks[:num_nodes]
st.dataframe(networks)

chain_str = simulator.get_transition_string(length=chain_length)
chain_list = chain_str.split(" -> ")

st.write(f"Simulation d'une suite de parcours de taille {chain_length}")
st.dataframe(chain_list)

# ---- Cumulative Sums ----
st.subheader("Sommes cumulées et convergence")

unique_states = list(set(chain_list))
cumulative_data = np.zeros((len(unique_states), len(chain_list)))
for idx, state in enumerate(unique_states):
    mask = np.array([1 if x == state else 0 for x in chain_list])
    cumulative_data[idx] = np.cumsum(mask) / np.arange(1, len(mask) + 1)

df_cum = pd.DataFrame(cumulative_data.T, columns=unique_states)
df_cum["n"] = range(len(chain_list))

final_cumulative_probs = df_cum.iloc[-1][df_cum.columns[:-1]]

if st.checkbox("Afficher la matrice des sommes cumulées", value=False):
    st.dataframe(df_cum)
    st.write("Probabilité stationnaire (méthode 2):")
    st.dataframe(final_cumulative_probs)

st.write("""Pour un grand nombre de simulations, le théorème d'ergodicité
    (loi des grands nombres des chaînes de Markov) nous assure la convergence
    vers la probabilité stationnaire de chaque page.""")

fig_conv = px.line(
    df_cum,
    x="n",
    y=df_cum.columns,
    title="Convergence des Probabilités",
)
st.plotly_chart(fig_conv, width="stretch")

# ---- Compare Methods ----
st.subheader("Comparaison entre les deux probabilités stationnaires")
col1, col2 = st.columns(2)
with col1:
    st.write("**Méthode 1** (puissances de matrice)")
    st.dataframe(stationary_prob)
with col2:
    st.write("**Méthode 2** (simulation de chaîne)")
    st.dataframe(final_cumulative_probs)

st.write(
    "On peut aisément voir que la différence entre les estimations est très petite."
)

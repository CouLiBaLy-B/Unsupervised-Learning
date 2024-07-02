import streamlit as st
from src.pagerank import PageRankSimulator


st.markdown(
    """<h1 style='text-align: center; background-color: #2D3E50; color: #FFFFFF'>
    Page Rank Simulator
    </h1>""",
    unsafe_allow_html=True,
)

"""
Simulation d'un graphe orienté de N sommets et de probabilité p.
Pour cette simulation, nous allons utiliser la librairie $networkx$ de python
que est l'quivalent du package R $igraph$.
"""
# Utilisation de la classe
simulator = PageRankSimulator()
d = simulator.param()
graph = simulator.generate_graph()
transition_matrix = simulator.generate_transition_matrix()
stationary_prob = simulator.calculate_stationary_probability()
simulation_size = st.number_input("Taille de la simulation :", min_value=10, max_value=10000, value=1000)
simulation_result = simulator.simulation(simulation_size)
cumulative_sums = simulator.generate_cumulative_sums(simulation_result.split(" -> "))
convergence_plot = simulator.plot_convergence(cumulative_sums)

# Affichage des résultats
st.write("Graphe simulé")
st.pyplot(graph)

# st.write("Matrice de transition")
r""" Une fois le graghe généré, nous pouvons facilement avoir
accès à la matrice d'adjacence à l'aide de adjacency_matrix
de networkx donc a la matrice de transition 
en ajoutant quelques transformations à celle-ci.
"""
'### La matrice de transition pour un $ϵ$ donné'
st.dataframe(transition_matrix)

# st.write("Probabilité stationnaire")
'### La probabilité stationnaire'
r"""Notre méthode de calcul de la probabilité stationnaire est d'utiliser les puissances de la
matrice de transition. En effet, en utilisant la définition de la probabilité stationnaire π = πA et par 
reccurence π = π$A^n$ pour n assez grand, nous obtenons un résultat  assez précis de la 
probabilité stationnaire."""
st.dataframe(stationary_prob)

'''### Simulation d'une chaine de Markov '''
st.write(f"Simulation d'une suite de parcours de taille {simulation_size}")
'''
Pour mettre dans un exemple concret, nous allons utiliser les reseaux sociaux les plus connus à savoir :'''
st.dataframe(simulation_result.split(" -> "))
st.write("### Convergence des Probabilités")

"""
Pour un grand nombre de simulation, le théorème d'ergodicité
(loi des grands nombres des chaines de Markov) 
nous assure la converge vers la probabilité de chaque page.
"""
st.plotly_chart(convergence_plot)

"""On aisément voir que la différence entre les estimations est très petite """
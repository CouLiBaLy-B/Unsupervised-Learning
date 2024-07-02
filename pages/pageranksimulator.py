import streamlit as st
from src.pagerank import PageRankSimulator


st.markdown(
    """<h1 style='text-align: center; background-color: #2D3E50; color: #FFFFFF'>
    Page Rank Simulator
    </h1>""",
    unsafe_allow_html=True,
)

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
st.write("Matrice de transition")
st.dataframe(transition_matrix)
st.write("Probabilité stationnaire")
st.dataframe(stationary_prob)
st.write(f"Simulation d'une suite de parcours de taille {simulation_size}")
st.dataframe(simulation_result.split(" -> "))
st.write("Convergence des Probabilités")
st.plotly_chart(convergence_plot)
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from networkx.generators.random_graphs import fast_gnp_random_graph as nxf


class PageRankSimulator:
    def __init__(self):
        self.d = None
        self.g = None
        self.MatriceA = None
        self.l = ["Facebook", "Whatsapp", "Instagram", "Twitter", "Teams", "Discord", "LinkedIn", "Snap", "Bloomberg", "slack", "Github"]
        
    def param(self):
        K = st.slider('Le nombre de sommet ', 7, 10, 8)
        p = st.slider('La probabilté p', 0.4, 1.0, 0.5)
        self.d = [K, p]
        return self.d

    def generate_graph(self):
        self.g = nxf(self.d[0], self.d[1], directed=True, seed=20222023)
        pos = nx.spring_layout(self.g)
        gp, ax = plt.subplots()
        ax = nx.draw(self.g, pos=pos, ax=ax, node_size=800, node_color='blue', edge_color='red', with_labels=True)
        return gp

    def eps(self):
        return st.slider('La probabilté  ϵ', 0.0, 0.50, 0.05)

    def generate_transition_matrix(self):
        mat = nx.adjacency_matrix(self.g)
        adj = np.zeros((self.d[0], self.d[0]), dtype=int)
        for i in range(self.d[0]):
            for j in range(self.d[0]):
                adj[i,j] = mat[i,j]

        Adj = pd.DataFrame(adj)
        ep = self.eps()
        MatriceA = pd.DataFrame(np.diag([ep for i in range(self.d[0])]))
        OtMat = Adj.apply(lambda x : (x-ep)/sum(x), axis=1)
        OtMat[OtMat < 0] = 0
        Matrice = MatriceA + OtMat
        self.MatriceA = np.array(Matrice)
        return Matrice

    def calculate_stationary_probability(self):
        i = st.number_input("La puissance n de la matrice de transition pour notre calcul :", min_value=10, max_value=10000, value=1000)
        stat1 = np.linalg.matrix_power(self.MatriceA, i)[0]
        return pd.Series(stat1, index=self.l[:self.d[0]])

    def simulation(self, n):
        k = np.random.choice(self.l[:self.d[0]])
        Chaine = k
        for i in range(n):
            k = np.random.choice(self.l[:self.d[0]], p=self.MatriceA[self.l[:self.d[0]].index(k)])
            Chaine += " -> " + k
        return Chaine

    def generate_cumulative_sums(self, liste):
        SomCum = np.zeros((len(np.unique(liste)), len(liste)))
        for i in range(len(self.l[:self.d[0]])):
            SomCum[i] = np.cumsum(self.recodage(self.l[i], liste))/range(1,len(self.recodage(self.l[i], liste))+1)
        df = pd.DataFrame(data=SomCum.T, columns=self.l[:self.d[0]])
        df["n"] = range(0, len(liste))
        return df

    @staticmethod
    def recodage(var, liste):
        return [1 if x == str(var) else 0 for x in liste]

    def plot_convergence(self, df):
        fig = px.line(df, x="n", y=df.columns, title="Convergence des Probabilités")
        return fig

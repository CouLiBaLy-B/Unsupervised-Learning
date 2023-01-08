import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import streamlit as st
from PIL import Image
from networkx.generators.random_graphs import fast_gnp_random_graph as nxf
import scipy as sc
st.set_page_config(
    page_title="Unsupervised_Learning",
    page_icon="😎",
    layout="wide"
)
'''
# Projet d'unsupervised Learning
'''
"## Bourahima COULIBALY "
"## M2 Data Science"
"## Université Paris Saclay"

image = Image.open('unsupervised.jpeg')
st.image(image)
"""
Cette application web a été créer avec la librairie Streamlit de python dans l'objectif de présenter et d'utiliser les résultats du projet unsupervised learning 1

"""

""" L'objetif de ce projet est d'appliqué les différents conceptes vue durant le cours de Unsupervised learning qui sont entre autres :

- Chaine de Markov et Chaine de Markov cachée pour les simulations de communauté

- Algorithme de Baum-Welch pour l'estimation et Viterbi pour une simulation

- SBM (Stochastic Block Model) pour la simulation également de communauté web

Pour ce qui concerne la rédaction, nous allons suivre la même trame que les questions dans le sujet du projet et apporter un peut de généralisation
sur les choix des différents paramètres fixés dans le sujet.
"""

'# 1. Algorithme Page Rank'
"""
Simulation d'un graphe orienté de N sommets et de probabilité p. Pour cette simulation, nous allons utiliser la librairie $networkx$ de python
que est l'quivalent du package R $igraph$.
"""

def param():
    K = st.slider('Le nombre de sommet ',7,10,8)
    p = st.slider('La probabilté p',0.4,1.0,0.5)
    d = [K, p]
    return d
d = param()
g = nxf(d[0], d[1], directed = True, seed =20222023)
pos = nx.spring_layout(g)
gp, ax = plt.subplots()
ax = nx.draw(g,pos = pos, ax =ax ,node_size=800,node_color='blue', edge_color='red',with_labels=True)
"Graphe simulé"
gp

""" Une fois le graghe généré, nous pouvons facilement avoir accès à la matrice d'adjacence à l'aide de $adjacency_matrix$ de $networkx$ donc a la matrice de transition 
en ajoutant quelques transformations à celle-ci.
"""
'### La matrice de transition pour un $ϵ$ donné'
def eps():
    eps = st.slider('La probabilté  ϵ', 0.0, 0.50, 0.05)
    return eps

#Matrice d'adjacence

mat = nx.adjacency_matrix(g)
adj = np.zeros((d[0],d[0]), dtype = int)
for i in range(d[0]):
    for j in range(d[0]):
        adj[i,j] = mat[i,j]

Adj = pd.DataFrame(adj)
ep  = eps()
MatriceA = pd.DataFrame(np.diag([ep for i in range(d[0])]))
OtMat = Adj.apply(lambda x : (x-ep)/sum(x)  , axis=1 )
OtMat[OtMat < 0] = 0
Matrice = MatriceA + OtMat
MatriceA = np.array(Matrice)

# Matrice de transition

Matrice

'### La probabilité stationnaire'
"""Notre méthode de calcul de la probabilité stationnaire est d'utiliser les puissances de la
matrice de transition. En effet, en utilisant la définition de la probabilité stationnaire π = πA et par 
reccurence π = π$A^n$ pour n assez grand, nous obtenons un résultat  assez précis de la 
probabilité stationnaire."""
def i():
    i = st.number_input("La puissance n de la matrice de transition pour notre calcul :",min_value=10,max_value=10000, value=1000)
    return i
i = i()
stat1 = np.linalg.matrix_power(MatriceA, i)[0]

stat1
'''### Simulation d'une chaine de Markov 

Pour mettre dans un exemple concret, nous allons utiliser les reseaux sociaux les plus connus à savoir :'''
l = ["Facebook","Whatsapp","Instagram","Twitter","Teams","Discord","LinkedIn","Snap","Bloomberg","slack","Github"]
st.dataframe(l)

stat1 = pd.Series(stat1, index=l[:d[0]])

def simulation(n,l , Matrice ):
  ''' Simulation :
  Cette fonction permet de generer une chaine de la Markov de longueur n, de matrice de transition Matrice et d'états possibles listés dans L.

  Parametres :
      n : la taille de la simulation
      l : Liste des états possible de la chiane
      Matrice : La matrice transition de la chiane de Markov '''
  k = np.random.choice(l)
  K =[l.index(k)]
  Chaine = k
  for i in range(n):
    k = np.random.choice(l, p = Matrice[l.index(k)])
    Chaine += " -> " + k
  return Chaine

def taille():
    i = st.number_input("Taille de la simulation :",min_value=10,max_value=10000, value=1000)
    return i
i = taille()
m = simulation(i, l[:d[0]], MatriceA)
liste = m.split(" -> ")
"Simulation d'une suite de parcours de taille ",i
st.dataframe(liste)

def recodage(var, liste =liste) :
  lst =[]
  for i in range(len(liste)):
    if liste[i] == str(var):
      lst.append(1)
    else :
      lst.append(0)
  return lst

SomCum = np.zeros((len(np.unique(liste)), len(liste)))
for i in range(len(l[:d[0]])):
  SomCum[i] = np.cumsum(recodage(l[i], liste))/range(1,len(recodage(l[i], liste))+1)
df = pd.DataFrame(data = SomCum.T, columns = l[:d[0]])
df["n"] = range(0,len(liste))
'Pour étudier le caractère ergodique (loi de notre chaine, '
stat2 = df.iloc[-1,:]
stat2.columns = "Proba Stationnaire"
if st.checkbox('Matrice des sommes cumulées', value=False):
    df
    "Probabilité stationnaire"
    stat2

"""
Pour un grand nombre de simulation, le théorème d'ergodicité (loi des grands nombres des chaines de Markov) nous assure la converge vers la probabilité de chaque page.
"""
fig, ax = plt.subplots()
fig = px.line(df,x = "n", y = df.columns,
              title = "Convergence des Probabilités"
              )
#fig.select_yaxes

fig
plt.show()

"""#### Comparaison entre les deux probabilité stationnaire"""
stat1.columns = "Proba Stationnaire"
col1, col2 = st.columns(2)
with col1:
    "Méthode 1"
    stat1
with col2:
    "Méthode 2"
    stat2[:-1]

"""On aisément voir que la différence entre les estimations est très petite """

'# 2. COMMUNICATION WEB'
'## Parcours web'
""" Dans cette section, nous allons simuler un parcours web d'une personne lambda à l'aide d'une chaîne de Markov cachée (HMM) dont les paramètres sont :
"""
" Etats cachés "
domaine = ["Sport","Culture","Beauté"]
st.dataframe(domaine)
A = np.array([0.7,0.2,0.1,0.25,0.7,0.05,0.1,0.1,0.8], dtype = float).reshape((3,3)) # Matrice de transition de la chaine de Markov

Mots = ["Abdominaux", "Cosmétiques","Livres","Age","Force", "Endurance","Résilience", "Crème", "Histoire", "Mathématiques"] # Matrice de changement des états cachés
B = np.array([0.2,0,0.1,
              0,0.1,0.3,
              0.1,0.3,0,
              0.1,0.2,0.2,
              0.2,0,0.1,
              0.3,0,0,
              0.1,0.1,0.1,
              0,0,0.2,
              0,0.2,0,
              0,0.1,0], dtype = float).reshape((10,3)).T
'La Matrice  de transition A'
A
" Etats visibles "
st.dataframe(Mots)
"La matrice d'emission B"
B
'''Pour faire la simulation à double mots clés, nous allons commencer par determiner la matrice des probabilités pour deux mots
ainsi que des couples de mots clés sans tenir compte ni de l'ordre. Ainsi dont la creation de notre matrice de probabilité sera sous condition de 
d'independance des deu mots clés par rapport au domaine est :

$$ D_{i,j} = P(X_j = W1, Y_j = W2 |Dom_i) = P(X_j = W1 |Dom_i) * P(Y_j = W2 |Dom_i)$$

qui provient de l'independance des deux mots clés '''

keyword = []
for i in Mots :
  for j in Mots :
    if (i,j) not in keyword and (j,i) not in keyword:
        keyword.append((i,j))
st.dataframe(keyword)

somme = np.sum(range(len(Mots)+1))
D = np.zeros((A.shape[0], somme))
for i in range(B.shape[0]):
  for j in range(B.shape[1]):
    for k in range(j, B.shape[1]):
      if j !=k :
        D[i][np.sum(range(B.shape[1]))- np.sum(range(B.shape[1]-j))+k] = 2*B[i][j]* B[i][k]
      else :
        D[i][np.sum(range(B.shape[1]))- np.sum(range(B.shape[1]-j))+k] = B[i][j]* B[i][k]
'La nouvelle matrice des probabilités est : ',D
'Vérification du caractère stochastique de la matrice', D.sum(axis = 1)

class Markov_caché :
  '''
    Objectif : Simulation d'une suite de chaîne de Markov
    Paramètre :
      - n : taille de la simulation
      - A : Matrice de transition
      - B : matrice d'émission
      - m : notre de répétition
  '''
  def __init__(self,n,A,B,m):
    self.n = n
    self.A = A
    self.B = B
    self.m = m
#---------------------------------------------------------------
  def simulationHMM(self):
    ''' Simulation :
    Cette fonction permet de générer une chaine de la Markov cachée de longueur n, de matrice de transition Matrice et d'états possibles listés dans L.

    Parametres :
        n : la taille de la simulation
        B : Matrice de changement des états cachés
        A : La matrice transition de la chiane de Markov   '''
    M = self.B.shape[0]
    k = self.B.shape[1]
    Z = np.zeros(self.n, dtype = int)
    X = np.zeros(self.n, dtype = int)
    Z[0] = np.random.choice(range(M))
    X[0] = np.random.choice(range(k), p = self.B[Z[0],])
    for i in range(1,self.n):
      Z[i] = np.random.choice(range(M),1, p = self.A[Z[i-1],])
      X[i] = np.random.choice(range(k), p = self.B[Z[i],])
    return np.array([Z,X])
#----------------------------------------------------------------
  def Mul_HMM(self):
    global domaine
    global Mots
    df = pd.DataFrame(Markov_caché.simulationHMM(self).T, columns = ["Domaines_0","Mots clefs_0"])
    df["Domaines_0"] = df["Domaines_0"].apply(lambda x : domaine[x])
    df["Mots clefs_0"] = df["Mots clefs_0"].apply(lambda x : keyword[x])
    i = 1
    while i < self.m :
      df["Domaines_"+str(i)] , df["Mots clefs_"+str(i)] = Markov_caché.simulationHMM(self)
      df["Domaines_"+str(i)] = df["Domaines_"+str(i)].apply(lambda x : domaine[x])
      df["Mots clefs_"+str(i)] = df["Mots clefs_"+str(i)].apply(lambda x : keyword[x])
      i = i+1
    return df

'Paramètre de la taille de la simulation de notre communauté'
def nbre():
    n = st.number_input("taille de la simulation", min_value= 10, max_value= 200,value=100)
    m = st.number_input("largeur de la simulation", min_value=1, max_value=100, value=30)
    return n,m
n,m = nbre()
f' Simulation de la chaine de Markov cachée pour deux mots clés de {m} parcours de longueur {n}'
M = Markov_caché(n = n,A = A,B = D,m =m)
Mat = M.Mul_HMM()
Mat



class Baum_welch:
    ''' Objectif : calcul des estimateurs de Baum Welch
    Paramètre :
        - V : observations (visible)
        - a : Matrice de transition
        - b : matrice d'emision
        - n_iter : nombre d'iteration maximal
        - initial : probabilité initiale'''
    def __init__(self, V, a,b,initial, n_iter):
        self.V = V
        self.a = a
        self.b = b
        self.initial = initial
        self.n_iter = n_iter

    def forward(self):
        alpha = np.zeros((self.V.shape[0], self.a.shape[0]))
        alpha[0, :] = self.initial* self.b[:, V[0]]
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
            alpha = Baum_welch.forward(self)
            beta = Baum_welch.backward(self)
            xi = np.zeros((M, M, T - 1))
            for t in range(T - 1):
                denominator = alpha[t, :].T @ self.a * self.b[:, self.V[t + 1]].T @ beta[t + 1, :]
                for i in range(M):
                    numerator = alpha[t, i] * self.a[i, :] * self.b[:, V[t + 1]].T * beta[t + 1, :].T
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


def Mul_baum_welch(df, A = A, D = D):
    l = df.shape[1]
    initial_distribution = np.ones(A.shape[0]) / A.shape[0]
    EST = []
    for i in range(1,l+1, 2) :
        V = df.iloc[: ,i].values
        Ba = Baum_welch(V=V, a=A, b=D, initial=initial_distribution)
        est = Ba.baum_welch()
        EST.append(est)
        if est["b"].sum(axis = 1)[0] != 1.0 :
            #est["b"] = pd.DataFrame(est["b"]).apply(lambda x: x/sum(x), axis = 1).values
            est["b"] = est["b"]/np.sum(est["b"], axis = 1).reshape((-1,1))
            est["a"] = est["a"] / np.sum(est["a"], axis=1).reshape((-1, 1))
        else :
            A,D = est["a"], est["b"]
    return {"a":A,"b":D, "i": i, "EST":EST}

#'### Recodage en fonction de leur index dans domaine et keyword'
mat = Mat.copy()
for i in range(mat.shape[0]):
    for j in range(0,mat.shape[1],2):
        mat.iloc[i,j] = domaine.index(Mat.iloc[i,j])
        mat.iloc[i,j+1] = keyword.index(Mat.iloc[i, j+1])


'''Pour faire le calculs des estimateurs par l'algorithm de Baum welch, il faut un récodage
des observations en entiers. Nous allons utiliser comme recodage l'indixe de 
chaque élément dans sa liste correspondante. Ainsi nous obtenons le tableau suivant'''
####
if st.checkbox("Voulez-vous voir la matrice recodée ?", value= False):
    mat

initial_distribution = np.ones(A.shape[0])/A.shape[0]
V = mat["Mots clefs_0"].values

Ba = Baum_welch(V =V,a = A,b = D, initial=initial_distribution, n_iter = 100)
est = Ba.baum_welch()
" Les estimateurs de Baum Welch de A et B sont :" 
"L'estimateur de la matrice A est : ",est["a"]
"L'estimateur de la matrice B est : ",est["b"]


#"Deuxieme appel"
#V = mat["Mots clefs_2"].values

#Ba = Baum_welch(V =V,a = est["a"],b = est["b"], initial=initial_distribution)
#est = Ba.baum_welch()
#"L'estimateur de la matrice A est : ",est["a"]
#"L'estimateur de la matrice B est : ",est["b"]
####
#"Mul baum welch "

#Baum = Mul_baum_welch(mat, A = A, D = D)
#Baum["EST"][0]["a"]
#Baum["EST"][0]["b"]
#Baum["EST"]
#Baum["i"]
####

def viterbi(V, a, b, initial_distribution):
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
            result.append(domaine[int(s)])
    return result

Verdi = viterbi(V, est["a"],est["b"], initial_distribution)
'### Générateur des états cachés en utilisant Viterbi'
"Cette simulation des états avec Viterbi utilisée comme paramètre les estimateurs obtinus par Baum Welch et les résultat est le suivant "
st.dataframe(Verdi)


"""L'algorithme de Viterbi (VA) est une solution optimale au sens du maximum de vraisemblance pour l’estimation d’une séquence d’états 
d’un processus de Markov à temps discrets et nombres d’états finis observés dans un bruit sans mémoire. Nous pouvons donc dire que le 
procèssus est le chemin optimal pour un suffeur au sens du maimum de vraisemble"""




'''# 3. Simulation des communauté web
Supposons qu'un ensemble n = 90 pages web soit partagé en 3 groupes:

• les pages traitant de sport (z = S)

• les pages traitant de culture (z = C)

• les pages traitant de soins beauté (z = B)

selon le processus suisvant : 

'''

r'''

• $ X_{ij}|z_{i} =k,z_{j} =l \sim B(\alpha I_{(k=l)} + \beta I_{(k = l)})$

• $∀k , P(z_i = k) = \pi_k = 1/3$

Avec $\alpha $ et $\beta$ ajustables.

'''
class SBM:
    def __init__(self, n, alpha, beta):
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def indicatrice(l,k):
        return 1 if l == k else 0

    def simulation_sbm(self):
        matrice = np.zeros((3,3), dtype=int)
        Z1 = [np.random.choice(["S","C","B"])]
        Z2 = [np.random.choice(["S","C","B"])]
        X = [np.random.binomial(1,self.alpha*SBM.indicatrice(Z1,Z2) + self.beta*(1+ SBM.indicatrice(Z1,Z2)))]
        for i in range(n):
            Z1.append(np.random.choice(["S","C","B"]))
            Z2.append(np.random.choice(["S", "C", "B"]))
            X.append(np.random.binomial(1,self.alpha* SBM.indicatrice(Z1,Z2) + self.beta*(1+ SBM.indicatrice(Z1,Z2))))
        for i in range(len(X)):
            if X[i] == 1:
                matrice[["S","C","B"].index(Z1[i]), ["S","C","B"].index(Z2[i])] = 1
        return {"Matrice" : matrice, "Z1": Z1, "Z2":Z2}


def parametres():
    n = st.number_input("La taille n de la simulation :", value= 90, min_value=10, max_value=1000)
    alpha = st.slider(r"Alpha $\alpha$ : ", min_value= 0.01,value=0.15, max_value=0.50, key = "alpha")
    beta = st.slider("Beta : ", min_value= 0.01, max_value=0.50, value=0.05, key = "beta")
    return n,alpha,beta

n, alpha,beta = parametres()

# Simulation
Sbm = SBM(n = n, alpha = alpha, beta = beta)

Simu = Sbm.simulation_sbm()
"La matrice d'adjacence X"
X = Simu["Matrice"]
X
if st.checkbox(r"""Voulez-vous voir Zi ?""", value=False):
    col1 , col2 = st.columns(2)
    with col1 :
        Simu["Z1"]
    with col2 :
        Simu["Z2"]





def eps():
    ep = st.number_input("Epsillon  = 1 / " , min_value=100, max_value=10000, value=1000)
    return ep
ep = 1/eps()

# Calcul de la matrice A

A1 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        A1[i,j] = (X[i,j]+ ep)/(np.sum(X[i,])+ n*ep)
"La matrice $A^1$"
A1

"La matrice $A^2$"
A2 = np.ones(X.shape)/np.sqrt(n)

A2

# Simulation





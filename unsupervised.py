import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
from tensorflow.keras.datasets import imdb
#from tensorflow.keras import utils

from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Activation, Embedding, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector

from tensorflow.keras.optimizers import Adam
import streamlit as st
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
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
"## Ibrahim COULIBALY "
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

sbm_d = D.copy()

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

Verdi = viterbi(V, A,D, initial_distribution)
'### Générateur des états cachés en utilisant Viterbi'
"Cette simulation des états avec Viterbi utilisée comme paramètre A et D et le résultat est le suivant "
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


"# Simulation"

def parametres():
    n = st.number_input("La taille n de la simulation :", value= 90, min_value=10, max_value=1000)
    alpha = st.slider(r"Alpha $\alpha$ : ", min_value= 0.01,value=0.15, max_value=0.50, key = "alpha")
    beta = st.slider("Beta : ", min_value= 0.01, max_value=0.50, value=0.05, key = "beta")
    return n,alpha,beta

n, alpha,beta = parametres()

def sbm_simulation(n = n, pi = [1/3, 1/3, 1/3], alpha = alpha, beta = beta):
  X = np.zeros((n,n))
  mm = np.random.multinomial(n, pvals = [1/3, 1/3, 1/3])
  Z = np.concatenate((np.concatenate((np.ones(mm[0]), 2*np.ones(mm[1]))), 3*np.ones(mm[2])))
  for i in range(n):
    for j in range(n):
      if Z[i] == Z[j]:
          p = alpha
      else:
          p = beta
      if (np.random.binomial(1, p) & (i != j)):
        X[i,j] = 1
  return Z, X

Z, X = sbm_simulation(n = n, pi = [1/3, 1/3, 1/3], alpha = alpha, beta = beta)

col1, col2 = st.columns(2)
with col1:
    "Les domaines"
    Z
with col2:
    "Matrice d'adjacence X"
    X

D = sbm_d

' Génération des mots visibles connaissant les mots caché'

def simulation_sbm(keyword, B, Z):
    Mots = []
    M = B.shape[1]
    for i in Z:
        Mots.append(keyword[np.random.choice(range(M), p = B[int(i- 1),])])
    return Mots
mots = simulation_sbm(keyword, sbm_d, Z)


col1, col2 = st.columns(2)
with col1:
    "Domaines (mots cachés)"
    Z
with col2:
    "Mots visibles"
    st.dataframe(mots)

# Calcul de la matrice A
def eps():
    ep = st.number_input("Epsillon  = 1 / " , min_value=100, max_value=10000, value=1000)
    return ep
ep = 1/eps()

r"$$ A^{1}_{ij} = \frac{X_{ij} + ϵ}{\sum_j X_{ij} + n ϵ} $$"

A1 = (X + ep)/(np.sum(X, axis = 1) + n * ep)[:,None]
if st.checkbox(r"Voulez vous afficher la matrice $A^1$ ?", value=False):
    r"La Matrice $A^1$", A1

r"$$ A^{2}_{ij} = \frac{1}{n}$$"
A2 = np.ones(X.shape) / n
if st.checkbox(r"Voulez vous afficher la matrice $A^2$ ?", value=False):
    r"La Matrice $A^2$", A2
if st.checkbox("Voulez vous vérifier le caractère stochastique de A1 et A2 ?", value=False):
    col1, col2 = st.columns(2)
    with col1:
        c = A1.sum(axis =1)
        c
    with col2:
        C = A2.sum(axis =1)
        C


def n():
    n = st.number_input("La taille la simulation :", min_value=100, max_value=10000, value=500)
    return n

n = n()

def MSimulation(A,Mots, n):
    m = A.shape[0]
    mot = np.random.choice(range(m))
    mots = []
    MOTS = [int(mot)/89]
    mots.append(Mots[mot])
    for i in range(1, n):
        mot = np.random.choice(range(m), p =A[mot, :])
        MOTS.append(int(mot)/89)
        mots.append(Mots[mot])
    return mots, MOTS

aa1, aa2 = MSimulation(A=A1, Mots = mots,  n=n)
AA1, AA2 = MSimulation(A= A2,Mots = mots, n=n)

col1, col2 = st.columns(2)
with col1:
    f"{n} simulations avec la matrice A1"
    st.dataframe(aa1)
with col2:
    f"{n} simulations avec la matrice A2"
    st.dataframe(AA1)



    
"### Classifier"
'''Utiliser un modèle de classification basé sur du deep learning
Le principe est le suivant :
 - Découper la chaine en batch de petite taille (c-a-d en petit segment de longueur definie)
 - A chaque batch on associe un label (0 ou 1) en fonction de la chaine (classe) de provenance
 - Que nous allons par la suite utiliser pour entrainé un modèle Perceptron multicouche (MLP) et d'un RNN'''


def decoupage(X1,X2, l_batch):
    X = []
    y = []
    for i in range(l_batch,(len(X1)-l_batch)):
        X.append(np.array(X1[(i-l_batch):i]))
        y.append(0)
    for i in range(l_batch,(len(X1)-l_batch)):
        X.append(np.array(X2[(i-l_batch):i]))
        y.append(1)
    return np.array(X), np.array(y)
def lot():
    l = st.number_input("La taille des lots (segment) :",min_value = 10, max_value = 200, value = 50, key = "lot")
    return l
l = lot()
X,y = decoupage(aa2, AA2, l)
if st.checkbox("Vu sur X et y", value = False):
    st.dataframe(X)
    st.dataframe(y)


r"""
### Méthode 1 :
### Objective:
Objectif :
Nous voulons implémenter un Perceptron multicouche (MLP) à deux couches avec 1 couche cachée en Python, pour un problème de classification.
La sortie du réseau est simplement la sortie de plusieurs fonctions en cascade :
- Transformations linéaires. On note les poids d'une transformation linéaire avec $W$ :

- Biais additifs. On note les paramètres des biais additifs avec $b$
- Les non-linéarités.
Pour cela, nous allons mettre en œuvre :

- propagation forward 
- calculer cost/loss
- propagation backward 
- mettre à jour les paramètre

De plus, nous définissons les grandeurs suivantes :

- $n^{[0]}$ : nombre de neurones d'entrée
- $n^{[1]}$ : nombre de neurones dans la couche cachée
- $n^{[2]}$ : nombre de neurones dans la couche de sortie
- $m$ : nombre de points de données d'apprentissage

Le **coût** est la moyenne de la **perte** sur les données d'apprentissage. 
Puisque nous avons affaire à un problème de classification binaire, nous utiliserons l'entropie croisée binaire.

$$\mathcal{L} = - \left( y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right),$$

où 
- les $y$ sont les étiquettes de vérité du sol des données et 
- les $\hat{y}$ sont les étiquettes estimées (sorties du réseau).

### Forward
$$
\newcommand{\l}[1]{^{[#1]}}
{Z\l1}{(m,n\l1)} = {X}{(m,n\l0)} {W\l1}{(n\l0,n\l1)}  + {b\l1}{n\l1} \\
{A\l1}{(m,n\l1)} = g\l1(Z\l1) \\
{Z\l2}{(m,n\l2)} = {A\l1}{(m,n\l1)} {W\l2}{(n\l1,n\l2)}  + {b\l2}{n\l2} \\
{A\l2}{(m,n\l2)} = \sigma(Z^{[2]})
$$

où 
- $g^{[1]}$ est une fonction d'activation non linéaire `Relu` (le code est fourni)
- $\sigma$ est une fonction d'activation de sortie sigmoïde (le code est fourni)


### Backward 

Backward propagation peut être comme suite : 

$$
\newcommand{\ddd}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\L}[0]{\mathcal{L}}
\newcommand{\l}[1]{^{[#1]}}
\newcommand{\dZdeux}[0]{ { \ddd{\L}{Z\l2} }{ (m,n\l2)} }
\newcommand{\dWdeux}[0]{ { \ddd{\L}{W\l2} }{ (n\l1,n\l2)} }
\newcommand{\dbdeux}[0]{ { \ddd{\L}{b\l2} }{ (n\l2)} }
\newcommand{\dAun}[0]{ { \ddd{\L}{A\l1} }{ (m,n\l1)} }
\newcommand{\dZun}[0]{ { \ddd{\L}{Z\l1} }{ (m,n\l1)} }
\newcommand{\dWun}[0]{ { \ddd{\L}{W\l1} }{ (n\l0,n\l1)} }
\newcommand{\dbun}[0]{ { \ddd{\L}{b\l1} }{ (n\l1)} }
\\
\dZdeux = {A\l2}{(m,n\l2)} - {Y}{(m,n\l2)}\\
\dWdeux = \frac{1}{m} {{A\l1}{(m,n\l1)}}^{T} \dZdeux \\
\dbdeux = \frac{1}{m} \sum_{i=1}^{m} \dZdeux \\
\dAun = \dZdeux {{W\l2}{(n\l1,n\l2)}}^{T}\\
\dZun = \dAun \: \odot \: {g\l1}' ({Z\l1}{(m,n\l1)})\\
\dWun = \frac{1}{m} {{X}{(m,n^{[0]})}}^{T} \dZun \\
\dbun = \frac{1}{m} \sum_{i=1}^{m} \dZun
$$

Sur la base des formules précédentes, écrivez l'algorithme de rétropropagation correspondant.


### Mise à jour des paramètres

- Implémenter une **première version** dans laquelle les paramètres sont mis à jour en utilisant une **descente de gradient simple** :

$$
\newcommand{\ddd}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\L}[0]{\mathcal{L}}
W = W - \alpha \ddd{\L}{W}
$$


- Implémentez une **seconde version** dans laquelle les paramètres sont mis à jour en utilisant la **méthode du momentum** :

$$
\newcommand{\ddd}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\L}[0]{\mathcal{L}}
V_{dW}(t) = \beta V_{dW}(t-1) + (1-\beta) \ddd{\L}{W} \\
W(t) = W(t-1) - \alpha V_{dW}(t)
$$


"""
### Definition des fonctions

def F_standardize(X):
    """
    standardiser X, c'est-à-dire soustraire la moyenne (sur les données) et diviser par l'écart-type (sur les données)
    
    Paramètres
    ----------
    X : tableau np de taille (m, n_0)
        matrice contenant les données d'observation
    
    Retourne
    -------
    X : tableau np de taille (m, n_0)
        version normalisée de X
    """
    
    X -= np.mean(X, axis=0, keepdims=True) 
    X /= (np.std(X, axis=0, keepdims=True) + 1e-16)
    return X

def F_sigmoid(x):
   
    return 1 / (1 + np.exp(-x))

def F_relu(x):
    
    return x * (x > 0)

def F_dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
def F_computeCost(hat_y, y):
    m = y.shape[0]
    loss = -(y*np.log(hat_y) + (1-y)*np.log(1-hat_y))
    cost = np.sum(loss) / m
    return cost

def F_computeAccuracy(hat_y, y):
    m = y.shape[0]    
    class_y = np.copy(hat_y)
    class_y[class_y>=0.5]=1
    class_y[class_y<0.5]=0
    return np.sum(class_y==y) / m

X = F_standardize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)

n_0 = X_train.shape[1]
n_2 = 1

r"""# Définir la classe MLP avec les méthodes avant, arrière et de mise à jour.

Dans le code, nous désignerons 
- $\frac{\partial \mathcal{L}}{\partial W^{[1]}}$ par ``dW1``, 
- $\frac{\partial \mathcal{L}}{\partial b^{[1]}}$ par ``db1``, 
- $\frac{\partial \mathcal{L}}{\partial W^{[2]}}$ par ``dW2``, 
- $\frac{\partial \mathcal{L}}{\partial b^{[2]}}$ par ``db2``, 
- $\frac{\partial \mathcal{L}}{\partial Z^{[1]}}$ par ``dZ1``, 
- $\frac{\partial \mathcal{L}}{\partial A^{[1]}}$ par ``dA1``, 
- ...
"""

class C_MultiLayerPerceptron:
    """
    Une classe utilisée pour représenter un perceptron multicouche avec 1 couche cachée.

    ...

    Attributs
    ----------
    W1, b1, W2, b2 :
        poids et biais à apprendre
    Z1, A1, Z2, A2 :
        valeurs des neurones internes à utiliser pour la rétropropagation
    dW1, db1, dW2, db2, dZ1, dZ2 :
        dérivées partielles de la perte en fonction des paramètres.
        exemple : dW1 = dLoss/dW1
    VdW1, Vdb1, VdW2, Vdb2 :
        termes de momentum
    do_bin0_multi1 :
        détermine si nous résolvons un problème de classification binaire ou multi-classes.
        
    Méthodes
    -------
    propagation avant
    
    propagation arrière
    
    mettre_paramètres_à_jour
    
    """

    W1, b1, W2, b2 = [], [], [], []
    A0, Z1, A1, Z2, A2 = [], [], [], [], []
    dW1, db1, dW2, db2 = [], [], [], []   
    dZ1, dA1, dZ2 = [], [], []
    # --- for momentum
    VdW1, Vdb1, VdW2, Vdb2 = [], [], [], []     
    
    def __init__(self, n_0, n_1, n_2):
        self.W1 = np.random.randn(n_0, n_1) * 0.01
        self.b1 = np.zeros(shape=(1, n_1))
        self.W2 = np.random.randn(n_1, n_2) * 0.01
        self.b2 = np.zeros(shape=(1, n_2))        
        # --- for momentum
        self.VdW1 = np.zeros(shape=(n_0, n_1)) 
        self.Vdb1 = np.zeros(shape=(1, n_1))
        self.VdW2 = np.zeros(shape=(n_1, n_2))
        self.Vdb2 = np.zeros(shape=(1, n_2))
        return

    
    def __setattr__(self, attrName, val):
        if hasattr(self, attrName):
            self.__dict__[attrName] = val
        else:
            raise Exception("self.%s note part of the fields" % attrName)

            

    def M_forwardPropagation(self, X):
        """Propagation vers l'avant dans le MLP

        Paramètres
        ----------
        X : tableau numpy (m, n_0)
            données d'observation

        Retourner
        ------
        hat_y : tableau numpy (m, 1)
            valeur prédite par le MLP
        """

        self.A0 = X

        self.Z1 = self.A0 @ self.W1 + self.b1
        self.A1 = F_relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = F_sigmoid(self.Z2)

        hat_y = self.A2
       

        return hat_y


    def M_backwardPropagation(self, X, y):
        """Propagation vers l'arrière dans le MLP

        Paramètres
        ----------
        X : tableau numpy (m, n_0)
            données d'observation
        y : tableau numpy (m, 1)
            classe de vérité fondamentale à prédire
            
        """
        
        m = y.shape[0]

        self.dZ2 = self.A2 - y
        self.dW2 = (1/m) * (self.A1.T @ self.dZ2)
        self.db2 = (1/m) * np.sum(self.dZ2, axis = 0, keepdims = True)
        self.dA1 = self.dZ2 @ self.W2.T

        self.dZ1 = np.multiply(self.dA1,F_dRelu(self.Z1))
        self.dW1 = (1/m) * (self.A0.T @ self.dZ1)
        self.db1 = (1/m) * np.sum(self.dZ1, axis=0, keepdims = True)

        return

    
    def M_gradientDescent(self, alpha):
        """Mettre à jour les paramètres du réseau en utilisant la descente de gradient

        Paramètres
        ----------
        alpha : float scalar
            quantité de mise à jour à chaque étape de la descente de gradient
            
        """
        self.W1 = self.W1 - alpha * self.dW1 
        self.b1 = self.b1 - alpha * self.db1
        self.W2 = self.W2 - alpha * self.dW2
        self.b2 = self.b2 - alpha * self.db2
        return

    
    def M_momentum(self, alpha, beta):
        """Mettre à jour les paramètres du réseau en utilisant la méthode momentum

        Paramètres
        ----------
        alpha : float scalar
            quantité de mise à jour à chaque étape de la descente du gradient
        beta : float scalar
            terme de momentum 
        """  
        
        self.VdW1 = beta * self.VdW1 + (1 - beta) * self.dW1
        self.W1 = self.W1 - alpha * self.VdW1

        self.Vdb1 = beta * self.Vdb1 + (1 - beta) * self.db1
        self.b1 = self.b1 - alpha * self.Vdb1

        self.VdW2 = beta * self.VdW2 + (1 - beta) * self.dW2
        self.W2 = self.W2 - alpha * self.VdW2

        self.Vdb2 = beta * self.Vdb2 + (1 - beta) * self.db2
        self.b2 = self.b2 - alpha * self.Vdb2
     
        return
  
def parametre():
    col1, col2, col3 , col4 = st.columns(4)
    with col1:
        n1 = st.number_input("nombre de neurons cachés", min_value= 1, max_value=100, value=10)
    with col2:
        n2 = st.number_input("nombre d'epochs", min_value=10, max_value=10000, value=5000)
    with col3:
        n3 = st.number_input("learning rate", min_value=0.0, max_value=1.0, value=0.1)
    with col4:
        n4 = st.number_input("paramètres bêta pour le momentum", min_value=0.0, max_value=2.0, value=0.9)
    return n1,n2,n3, n4

n_1, nb_epoch,alpha,beta = parametre()


# Instancie la classe MLP en fournissant 
# la taille des différentes couches (n_0=n_input, n_1=n_hidden, n_2=n_output) 

myMLP = C_MultiLayerPerceptron(n_0, n_1, n_2)

train_cost, train_accuracy, test_cost, test_accuracy = [], [], [], []


for num_epoch in range(0, nb_epoch):
    
    # --- Forward
    hat_y_train = myMLP.M_forwardPropagation(X_train)
    
    # ---  resultat sur train
    train_cost.append( F_computeCost(hat_y_train, y_train) )
    train_accuracy.append( F_computeAccuracy(hat_y_train, y_train) )
    
    # --- Backward
    myMLP.M_backwardPropagation(X_train, y_train)
    
    # --- Mise à jour
    myMLP.M_gradientDescent(alpha)
    myMLP.M_momentum(alpha, beta)

    # --- resultat sur test
    hat_y_test = myMLP.M_forwardPropagation(X_test)
    test_cost.append( F_computeCost(hat_y_test, y_test) )    
    test_accuracy.append( F_computeAccuracy(hat_y_test, y_test) )
    
    if (num_epoch % 500)==0: 
        st.write("epoch: {0:d} (cost: train {1:.2f} test {2:.2f}) (accuracy: train {3:.2f} test {4:.2f})".format(num_epoch, train_cost[-1], test_cost[-1], train_accuracy[-1], test_accuracy[-1]))

col1,col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.plot(train_cost, 'r')
    ax.plot(test_cost, 'g--')
    plt.xlabel('# epoch')
    plt.ylabel('loss')
    plt.grid(True)
    fig
    plt.show()
with col2:
    fig, ax = plt.subplots()
    ax.plot(train_accuracy, 'r')
    ax.plot(test_accuracy, 'g--')
    plt.xlabel('# epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    fig
    plt.show()
    

K.clear_session()

"""### Deuxième méthode
Pour améliorer les performances du modèle, n'oubliez pas d'augmenter la quantité de donnée"""

# Decoupage en train et test de nos données 
m = int(n*0.2)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=m, random_state=42, shuffle=True, stratify=y)


k = st.checkbox("Lancer l'apprentissage du modèle du RNN", value = False)
if k:
   
    X_train = sequence.pad_sequences(x_train, maxlen=l)
    X_test = sequence.pad_sequences(x_test, maxlen=l)
  

# Creation et apprentissage du model
# CODE-RNN
if k:
    model = Sequential([
        Embedding(2,32 , input_length=l),
        Lambda(lambda x: K.mean(x, axis=1)),
        Dense(1),
        Activation('sigmoid')
    ])
    X = Input(shape=(x_train.shape[1],))
    Z = Embedding(2,32, input_length=l)(X)
    Z = Lambda(lambda x: K.mean(x, axis=1))(Z)
    Z = Dense(1)(Z)
    Y = Activation('softmax')(Z)
    model = Model(inputs=X, outputs=Y)
    model.summary()


    # --- compile and fit the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

    scores = model.evaluate(X_test, y_test, verbose=0)


    "Accuracy du modèle est :" , (scores[1]*100)

    hist.history.keys()

    fig, a = plt.subplots()
    plt.plot(hist.history['accuracy'], label='training set',marker='o', linestyle='solid',linewidth=1, markersize=6)
    plt.plot(hist.history['val_accuracy'], label='validation set',marker='o', linestyle='solid',linewidth=1, markersize=6)
    plt.title("model accuracy")
    plt.xlabel('#Epochs')
    plt.ylabel('Acuracy')
    plt.legend(bbox_to_anchor=( 1., 1.))
    fig
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(hist.history['loss'], label='training set',marker='o', linestyle='solid',linewidth=1, markersize=6)
    ax.plot(hist.history['val_loss'], label='validation set',marker='o', linestyle='solid',linewidth=1, markersize=6)
    plt.title("model loss")
    plt.xlabel('#Epochs')
    plt.ylabel('Total Loss')
    plt.legend(bbox_to_anchor=( 1.35, 1.))
    fig
    plt.show()

    y_pred = model.predict(x_test)

    y_true = y_test.argmax(1)

    y_pred = y_pred.argmax(1)

    cf_matri = confusion_matrix(y_true, y_pred)


    fig ,ax= plt.subplots()
    ax = sns.heatmap(cf_matri, annot = True)
    fig
    plt.show()

    
"# Fin merci pour votre lecture"

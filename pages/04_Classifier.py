"""Streamlit page - Classifier (MLP) Training."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.ml.classifier import (
    MultiLayerPerceptron,
    accuracy,
    binary_cross_entropy,
    prepare_data,
    split_into_batches,
    standardize,
)
from src.models.markov import MarkovModel, WebCommunitySimulator
from src.utils.config import (
    DEFAULT_DOMAINS,
    DEFAULT_EMISSION_MATRIX,
    DEFAULT_KEYWORDS,
)

st.set_page_config(
    page_title="Classifier - Unsupervised Learning",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    "<h1 style='text-align: center; background-color: #2D3E50; color: #FFFFFF'>"
    "4. Classification - Perceptron Multicouche</h1>",
    unsafe_allow_html=True,
)


def main() -> None:
    """Run the MLP classifier training app."""
    st.write("""Utiliser un modèle de classification basé sur du deep learning.

        Le principe est le suivant :
        - Découper la chaîne en batch de petite taille (segments de longueur définie)
        - À chaque batch on associe un label (0 ou 1) en fonction de la chaîne de provenance
        - Entraîner un modèle Perceptron Multicouche (MLP) avec descente de gradient et momentum
        """)

    # ---- Sidebar Parameters ----
    st.sidebar.header("Paramètres MLP")
    n_nodes_sbm = st.sidebar.number_input(
        "Taille SBM pour la génération des données",
        min_value=10,
        max_value=1000,
        value=90,
    )
    alpha_sbm = st.sidebar.slider("Alpha SBM", 0.01, 0.50, 0.15, key="alpha_sbm")
    beta_sbm = st.sidebar.slider("Beta SBM", 0.01, 0.50, 0.05, key="beta_sbm")

    n_hidden = st.sidebar.number_input(
        "Nombre de neurones cachés", min_value=1, max_value=100, value=10
    )
    n_epochs = st.sidebar.number_input(
        "Nombre d'epochs", min_value=10, max_value=10000, value=5000, step=500
    )
    learning_rate = st.sidebar.number_input(
        "Learning rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01
    )
    momentum_beta = st.sidebar.number_input(
        "Paramètre bêta pour le momentum", min_value=0.0, max_value=2.0, value=0.9
    )
    batch_size = st.sidebar.number_input(
        "Taille des segments", min_value=10, max_value=200, value=50
    )

    # ---- Data Generation ----
    st.header("Génération des données")

    if st.button("Générer les données"):
        with st.spinner("Génération des données..."):
            sbm = WebCommunitySimulator(
                n_nodes=n_nodes_sbm,
                alpha=alpha_sbm,
                beta=beta_sbm,
            )
            sbm.simulate()
            A1, A2 = sbm.compute_transition_matrices(1000)

            # Build emission model for observations during walks
            markov_model = MarkovModel(
                states=DEFAULT_DOMAINS,
                observations=DEFAULT_KEYWORDS,
                transition_matrix=np.array(
                    [[0.7, 0.2, 0.1], [0.25, 0.7, 0.05], [0.1, 0.1, 0.8]]
                ),
                emission_matrix=np.array(DEFAULT_EMISSION_MATRIX),
            )

            # Generate real random walks on A1 and A2 and encode node positions
            n_walk_steps = n_nodes_sbm * 5
            _, walk1_positions = sbm.simulate_random_walk(
                transition_matrix=A1,
                emission_matrix=markov_model.joint_emission_matrix,
                observation_pairs=markov_model.observation_pairs,
                n_steps=n_walk_steps,
            )
            _, walk2_positions = sbm.simulate_random_walk(
                transition_matrix=A2,
                emission_matrix=markov_model.joint_emission_matrix,
                observation_pairs=markov_model.observation_pairs,
                n_steps=n_walk_steps,
            )

        # Encode walks as node-position sequences for ML
        walk1_encoded = [int(p * (n_nodes_sbm - 1)) for p in walk1_positions]
        walk2_encoded = [int(p * (n_nodes_sbm - 1)) for p in walk2_positions]

        # Encode walks as integers for ML
        X_encoded, y_encoded = split_into_batches(
            sequence_1=walk1_encoded,
            sequence_2=walk2_encoded,
            batch_size=batch_size,
        )

        # Standardize and split
        X = standardize(X_encoded.astype(float))
        y = y_encoded.reshape(-1, 1)

        X_train, X_test, y_train, y_test = prepare_data(X, y)

        st.write(f"Taille de X : {X.shape}")
        st.write(f"Taille de y : {y.shape}")

        if st.checkbox("Afficher X et y", value=False):
            st.dataframe(X)
            st.dataframe(y)

        # ---- Train MLP ----
        st.header("Entraînement du MLP")

        if st.button("Lancer l'entraînement"):
            with st.spinner("Entraînement en cours..."):
                n_input = X_train.shape[1]
                n_output = 1

                mlp = MultiLayerPerceptron(n_input, n_hidden, n_output)

                train_cost, train_acc = [], []
                test_cost, test_acc = [], []

                for epoch in range(n_epochs):
                    # Forward
                    y_train_pred = mlp.forward(X_train)
                    train_cost.append(binary_cross_entropy(y_train_pred, y_train))
                    train_acc.append(accuracy(y_train_pred, y_train))

                    # Backward
                    mlp.backward(X_train, y_train)

                    # Update — use momentum if beta > 0, else simple gradient descent
                    if momentum_beta > 0:
                        mlp.update_momentum(learning_rate, momentum_beta)
                    else:
                        mlp.update_simple(learning_rate)

                    # Test metrics
                    y_test_pred = mlp.forward(X_test)
                    test_cost.append(binary_cross_entropy(y_test_pred, y_test))
                    test_acc.append(accuracy(y_test_pred, y_test))

                    if epoch % 500 == 0:
                        st.write(
                            f"epoch: {epoch} "
                            f"(cost: train {train_cost[-1]:.2f} test {test_cost[-1]:.2f}) "
                            f"(accuracy: train {train_acc[-1]:.2f} test {test_acc[-1]:.2f})"
                        )

            # ---- Plot Results ----
            st.header("Résultats")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(train_cost, "r", label="Train")
            ax1.plot(test_cost, "g--", label="Test")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(train_acc, "r", label="Train")
            ax2.plot(test_acc, "g--", label="Test")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True)

            st.pyplot(fig)

            st.write(
                """Comme on peut s'y attendre, les performances dépendent de la taille
                des segments. Plus ils sont longs, plus la précision est élevée."""
            )


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Lambda, Input
from keras.preprocessing import sequence
from keras import backend as K
from sklearn.metrics import confusion_matrix
import streamlit as st


class DataPreprocessor:
    @staticmethod
    def standardize(X):
        X -= np.mean(X, axis=0, keepdims=True)
        X /= (np.std(X, axis=0, keepdims=True) + 1e-16)
        return X

    @staticmethod
    def split_data(X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y)

    @staticmethod
    def pad_sequences(x_train, x_test, maxlen):
        X_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        return X_train, X_test

class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def dRelu(x):
        return (x > 0).astype(float)

class CostFunctions:
    @staticmethod
    def compute_cost(hat_y, y):
        m = y.shape[0]
        loss = -(y * np.log(hat_y) + (1-y) * np.log(1-hat_y))
        return np.sum(loss) / m

    @staticmethod
    def compute_accuracy(hat_y, y):
        m = y.shape[0]
        class_y = (hat_y >= 0.5).astype(int)
        return np.sum(class_y == y) / m

class MultiLayerPerceptron:
    def __init__(self, n_0, n_1, n_2):
        self.W1 = np.random.randn(n_0, n_1) * 0.01
        self.b1 = np.zeros(shape=(1, n_1))
        self.W2 = np.random.randn(n_1, n_2) * 0.01
        self.b2 = np.zeros(shape=(1, n_2))
        self.VdW1 = np.zeros(shape=(n_0, n_1))
        self.Vdb1 = np.zeros(shape=(1, n_1))
        self.VdW2 = np.zeros(shape=(n_1, n_2))
        self.Vdb2 = np.zeros(shape=(1, n_2))

    def forward_propagation(self, X):
        self.A0 = X
        self.Z1 = self.A0 @ self.W1 + self.b1
        self.A1 = ActivationFunctions.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = ActivationFunctions.sigmoid(self.Z2)
        return self.A2

    def backward_propagation(self, X, y):
        m = y.shape[0]
        self.dZ2 = self.A2 - y
        self.dW2 = (1/m) * (self.A1.T @ self.dZ2)
        self.db2 = (1/m) * np.sum(self.dZ2, axis=0, keepdims=True)
        self.dA1 = self.dZ2 @ self.W2.T
        self.dZ1 = np.multiply(self.dA1, ActivationFunctions.dRelu(self.Z1))
        self.dW1 = (1/m) * (self.A0.T @ self.dZ1)
        self.db1 = (1/m) * np.sum(self.dZ1, axis=0, keepdims=True)

    def gradient_descent(self, alpha):
        self.W1 -= alpha * self.dW1
        self.b1 -= alpha * self.db1
        self.W2 -= alpha * self.dW2
        self.b2 -= alpha * self.db2

    def momentum(self, alpha, beta):
        self.VdW1 = beta * self.VdW1 + (1 - beta) * self.dW1
        self.W1 -= alpha * self.VdW1
        self.Vdb1 = beta * self.Vdb1 + (1 - beta) * self.db1
        self.b1 -= alpha * self.Vdb1
        self.VdW2 = beta * self.VdW2 + (1 - beta) * self.dW2
        self.W2 -= alpha * self.VdW2
        self.Vdb2 = beta * self.Vdb2 + (1 - beta) * self.db2
        self.b2 -= alpha * self.Vdb2

class RNNModel:
    @staticmethod
    def create_model(input_length):
        model = Sequential([
            Embedding(2, 32, input_length=input_length),
            Lambda(lambda x: K.mean(x, axis=1)),
            Dense(1),
            Activation('sigmoid')
        ])
        return model

    @staticmethod
    def compile_and_fit(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=32):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

class Visualizer:
    @staticmethod
    def plot_training_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history.history['accuracy'], label='training set', marker='o', linestyle='solid', linewidth=1, markersize=6)
        ax1.plot(history.history['val_accuracy'], label='validation set', marker='o', linestyle='solid', linewidth=1, markersize=6)
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel('#Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(history.history['loss'], label='training set', marker='o', linestyle='solid', linewidth=1, markersize=6)
        ax2.plot(history.history['val_loss'], label='validation set', marker='o', linestyle='solid', linewidth=1, markersize=6)
        ax2.set_title("Model Loss")
        ax2.set_xlabel('#Epochs')
        ax2.set_ylabel('Total Loss')
        ax2.legend()

        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        cf_matrix = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cf_matrix, annot=True, ax=ax)
        ax.set_title("Confusion Matrix")
        return fig

class StreamlitApp:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.mlp = None
        self.rnn_model = None
        self.rnn_model = None
        self.X = None
        self.y = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        # Remplacez ceci par votre propre logique de chargement de données
        # Par exemple, vous pourriez utiliser:
        # self.X, self.y = load_your_data_function()
        # Ou pour cet exemple, nous allons simplement générer des données aléatoires
        self.X = np.random.rand(1000, 10)  # 1000 échantillons, 10 caractéristiques
        self.y = np.random.randint(2, size=(1000, 1))  # Classes binaires

    def run(self):
        st.title("Web Simulation avec Chaîne de Markov Cachée")
 
        # Chargez les données
        self.load_data()

        # Input parameters
        n_1 = st.number_input("Nombre de neurones cachés", min_value=1, max_value=100, value=10)
        nb_epoch = st.number_input("Nombre d'epochs", min_value=10, max_value=10000, value=5000)
        alpha = st.number_input("Learning rate", min_value=0.0, max_value=1.0, value=0.1)
        beta = st.number_input("Paramètre beta pour le momentum", min_value=0.0, max_value=2.0, value=0.9)

        # MLP Training
        if st.button("Entraîner MLP"):
            self.train_mlp(n_1, nb_epoch, alpha, beta)

        # RNN Training
        if st.checkbox("Lancer l'apprentissage du modèle RNN", value=False):
            self.train_rnn()

    def train_mlp(self, n_1, nb_epoch, alpha, beta):
        # Assume X, y, n_0, n_2 are defined elsewhere
        X = self.preprocessor.standardize(self.X)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, self.y)

        n_0 = X_train.shape[1]
        n_2 = 1
        self.mlp = MultiLayerPerceptron(n_0, n_1, n_2)

        train_cost, train_accuracy, test_cost, test_accuracy = [], [], [], []

        for num_epoch in range(nb_epoch):
            hat_y_train = self.mlp.forward_propagation(X_train)
            train_cost.append(CostFunctions.compute_cost(hat_y_train, y_train))
            train_accuracy.append(CostFunctions.compute_accuracy(hat_y_train, y_train))

            self.mlp.backward_propagation(X_train, y_train)
            self.mlp.gradient_descent(alpha)
            self.mlp.momentum(alpha, beta)

            hat_y_test = self.mlp.forward_propagation(X_test)
            test_cost.append(CostFunctions.compute_cost(hat_y_test, y_test))
            test_accuracy.append(CostFunctions.compute_accuracy(hat_y_test, y_test))

            if (num_epoch % 500) == 0:
                st.write(f"epoch: {num_epoch} (cost: train {train_cost[-1]:.2f} test {test_cost[-1]:.2f}) (accuracy: train {train_accuracy[-1]:.2f} test {test_accuracy[-1]:.2f})")

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(train_cost, 'r', label='Train')
        ax1.plot(test_cost, 'g--', label='Test')
        ax1.set_xlabel('# epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(train_accuracy, 'r', label='Train')
        ax2.plot(test_accuracy, 'g--', label='Test')
        ax2.set_xlabel('# epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig)

    def train_rnn(self):
        # Assume x_train, x_test, y_train, y_test, l are defined elsewhere
        X_train, X_test = self.preprocessor.pad_sequences(self.x_train, self.X_test, l)

        self.rnn_model = RNNModel.create_model(l)
        self.rnn_model.summary()

        history = RNNModel.compile_and_fit(self.rnn_model, X_train, y_train, X_test, y_test)

        scores = self.rnn_model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Accuracy du modèle est : {scores[1]*100}")

        # Plot training history
        fig = Visualizer.plot_training_history(history)
        st.pyplot(fig)

        # Plot confusion matrix
        y_pred = self.rnn_model.predict(X_test)
        y_true = y_test.argmax(1)
        y_pred = y_pred.argmax(1)
        fig = Visualizer.plot_confusion_matrix(y_true, y_pred)
        st.pyplot(fig)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
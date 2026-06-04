"""Machine Learning module for unsupervised classification.

Contains implementations of MLP and RNN models for sequence classification.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# Activation Functions
# ============================================================


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return x * (x > 0)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU activation function."""
    return (x > 0).astype(float)


# ============================================================
# Data Preprocessing
# ============================================================


def standardize(X: np.ndarray) -> np.ndarray:
    """Standardize features by removing the mean and scaling to unit variance.

    Args:
        X: Input data matrix (m, n_features)

    Returns:
        Standardized data matrix
    """
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-16
    return (X - mean) / std


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and test sets.

    Args:
        X: Feature matrix
        y: Target labels
        test_size: Fraction of data to use for testing
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )


# ============================================================
# Cost Functions
# ============================================================


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute binary cross-entropy loss.

    Args:
        y_pred: Predicted probabilities
        y_true: True labels

    Returns:
        Average loss value
    """
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_pred: Predicted probabilities
        y_true: True labels

    Returns:
        Accuracy score
    """
    y_pred_binary = (y_pred >= 0.5).astype(int)
    return np.mean(y_pred_binary == y_true)


# ============================================================
# Data Preparation for ML
# ============================================================


def split_into_batches(
    sequence_1: List[int],
    sequence_2: List[int],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split two sequences into batches for binary classification.

    Args:
        sequence_1: First sequence (class 0)
        sequence_2: Second sequence (class 1)
        batch_size: Size of each batch/segment

    Returns:
        Tuple of (X, y) where X is the batched features and y are the labels
    """
    X = []
    y = []

    # Batches from first sequence (class 0)
    for i in range(batch_size, len(sequence_1)):
        X.append(np.array(sequence_1[i - batch_size : i]))
        y.append(0)

    # Batches from second sequence (class 1)
    for i in range(batch_size, len(sequence_2)):
        X.append(np.array(sequence_2[i - batch_size : i]))
        y.append(1)

    return np.array(X), np.array(y)


# ============================================================
# Multilayer Perceptron
# ============================================================


class MultiLayerPerceptron:
    """A Multilayer Perceptron with one hidden layer.

    Supports both simple gradient descent and momentum-based updates.

    Args:
        n_input: Number of input neurons
        n_hidden: Number of hidden neurons
        n_output: Number of output neurons (1 for binary classification)
    """

    def __init__(self, n_input: int, n_hidden: int, n_output: int) -> None:
        # Initialize weights with small random values
        self.W1 = np.random.randn(n_input, n_hidden) * 0.01
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * 0.01
        self.b2 = np.zeros((1, n_output))

        # Momentum terms
        self.VdW1 = np.zeros_like(self.W1)
        self.Vdb1 = np.zeros_like(self.b1)
        self.VdW2 = np.zeros_like(self.W2)
        self.Vdb2 = np.zeros_like(self.b2)

        # Forward pass cache
        self.A0: np.ndarray | None = None
        self.Z1: np.ndarray | None = None
        self.A1: np.ndarray | None = None
        self.Z2: np.ndarray | None = None
        self.A2: np.ndarray | None = None

        # Backward pass cache
        self.dW1: np.ndarray | None = None
        self.db1: np.ndarray | None = None
        self.dW2: np.ndarray | None = None
        self.db2: np.ndarray | None = None
        self.dZ1: np.ndarray | None = None
        self.dZ2: np.ndarray | None = None
        self.dA1: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation through the network.

        Args:
            X: Input data (m, n_input)

        Returns:
            Output predictions (m, n_output)
        """
        self.A0 = X
        self.Z1 = self.A0 @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """Backward propagation to compute gradients.

        Args:
            X: Input data (m, n_input)
            y: True labels (m, n_output)
        """
        m = y.shape[0]

        # Output layer gradients
        self.dZ2 = self.A2 - y
        self.dW2 = (1 / m) * (self.A1.T @ self.dZ2)
        self.db2 = (1 / m) * np.sum(self.dZ2, axis=0, keepdims=True)
        self.dA1 = self.dZ2 @ self.W2.T

        # Hidden layer gradients
        self.dZ1 = np.multiply(self.dA1, relu_derivative(self.Z1))
        self.dW1 = (1 / m) * (self.A0.T @ self.dZ1)
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=0, keepdims=True)

    def update_simple(self, learning_rate: float) -> None:
        """Update parameters using simple gradient descent.

        Args:
            learning_rate: Step size for parameter updates
        """
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def update_momentum(self, learning_rate: float, beta: float) -> None:
        """Update parameters using momentum-based gradient descent.

        Args:
            learning_rate: Step size for parameter updates
            beta: Momentum coefficient
        """
        self.VdW1 = beta * self.VdW1 + (1 - beta) * self.dW1
        self.W1 -= learning_rate * self.VdW1
        self.Vdb1 = beta * self.Vdb1 + (1 - beta) * self.db1
        self.b1 -= learning_rate * self.Vdb1
        self.VdW2 = beta * self.VdW2 + (1 - beta) * self.dW2
        self.W2 -= learning_rate * self.VdW2
        self.Vdb2 = beta * self.Vdb2 + (1 - beta) * self.db2
        self.b2 -= learning_rate * self.Vdb2


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_hidden: int = 10,
    epochs: int = 5000,
    learning_rate: float = 0.1,
    beta: float = 0.9,
) -> Dict[str, List[float]]:
    """Train an MLP model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_hidden: Number of hidden neurons
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        beta: Momentum coefficient

    Returns:
        Dictionary containing training history metrics
    """
    n_input = X_train.shape[1]
    n_output = 1

    model = MultiLayerPerceptron(n_input, n_hidden, n_output)

    train_cost, train_accuracy = [], []
    test_cost, test_accuracy = [], []

    for _epoch in range(epochs):
        # Forward pass
        y_train_pred = model.forward(X_train)

        # Record metrics
        train_cost.append(binary_cross_entropy(y_train_pred, y_train))
        train_accuracy.append(accuracy(y_train_pred, y_train))

        # Backward pass
        model.backward(X_train, y_train)

        # Update parameters
        model.update_simple(learning_rate)
        model.update_momentum(learning_rate, beta)

        # Test metrics
        y_test_pred = model.forward(X_test)
        test_cost.append(binary_cross_entropy(y_test_pred, y_test))
        test_accuracy.append(accuracy(y_test_pred, y_test))

    return {
        "train_cost": train_cost,
        "train_accuracy": train_accuracy,
        "test_cost": test_cost,
        "test_accuracy": test_accuracy,
    }

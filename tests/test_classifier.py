"""Tests for the ML classifier module."""

import numpy as np
import pytest

from src.ml.classifier import (
    MultiLayerPerceptron,
    accuracy,
    binary_cross_entropy,
    prepare_data,
    split_into_batches,
    standardize,
)


class TestStandardize:
    """Test data standardization."""

    def test_mean_close_to_zero(self):
        """Test that standardized data has mean close to zero."""
        X = np.random.randn(100, 5) * 2 + 10
        X_std = standardize(X)
        means = np.mean(X_std, axis=0)
        np.testing.assert_allclose(means, 0, atol=1e-10)

    def test_std_close_to_one(self):
        """Test that standardized data has std close to one."""
        X = np.random.randn(100, 5) * 2 + 10
        X_std = standardize(X)
        stds = np.std(X_std, axis=0)
        np.testing.assert_allclose(stds, 1, atol=1e-6)


class TestPrepareData:
    """Test data splitting."""

    def test_split_shapes(self):
        """Test that split produces correct shapes."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = prepare_data(X, y, test_size=0.2)

        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20


class TestCostFunctions:
    """Test cost and accuracy functions."""

    def test_binary_cross_entropy_perfect(self):
        """Test BCE with perfect predictions."""
        y_pred = np.array([[0.99], [0.01], [0.99]])
        y_true = np.array([[1], [0], [1]])
        loss = binary_cross_entropy(y_pred, y_true)
        assert loss < 0.1

    def test_binary_cross_entropy_worst(self):
        """Test BCE with worst predictions."""
        y_pred = np.array([[0.01], [0.99], [0.01]])
        y_true = np.array([[1], [0], [1]])
        loss = binary_cross_entropy(y_pred, y_true)
        assert loss > 2.0

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_pred = np.array([[0.9], [0.1], [0.9]])
        y_true = np.array([[1], [0], [1]])
        assert accuracy(y_pred, y_true) == 1.0

    def test_accuracy_zero(self):
        """Test accuracy with all wrong predictions."""
        y_pred = np.array([[0.1], [0.9], [0.1]])
        y_true = np.array([[1], [0], [1]])
        assert accuracy(y_pred, y_true) == 0.0


class TestMultiLayerPerceptron:
    """Test MLP class."""

    @pytest.fixture
    def mlp(self):
        """Create a small MLP for testing."""
        return MultiLayerPerceptron(n_input=5, n_hidden=3, n_output=1)

    def test_forward_shape(self, mlp):
        """Test forward pass output shape."""
        X = np.random.randn(10, 5)
        output = mlp.forward(X)
        assert output.shape == (10, 1)

    def test_backward_shapes(self, mlp):
        """Test backward pass computes gradients of correct shapes."""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, (10, 1))

        mlp.forward(X)
        mlp.backward(X, y)

        assert mlp.dW1.shape == (5, 3)
        assert mlp.db1.shape == (1, 3)
        assert mlp.dW2.shape == (3, 1)
        assert mlp.db2.shape == (1, 1)

    def test_update_simple_changes_weights(self, mlp):
        """Test that simple gradient descent updates weights."""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, (10, 1))

        mlp.forward(X)
        mlp.backward(X, y)

        old_W1 = mlp.W1.copy()
        mlp.update_simple(learning_rate=0.01)
        assert not np.array_equal(mlp.W1, old_W1)

    def test_update_momentum_changes_weights(self, mlp):
        """Test that momentum updates change weights."""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, (10, 1))

        mlp.forward(X)
        mlp.backward(X, y)

        old_W1 = mlp.W1.copy()
        mlp.update_momentum(learning_rate=0.01, beta=0.9)
        assert not np.array_equal(mlp.W1, old_W1)

    def test_training_reduces_loss(self):
        """Test that training actually reduces loss."""
        np.random.seed(42)
        X_train = np.random.randn(200, 5)
        y_train = np.random.randint(0, 2, (200, 1))
        X_test = np.random.randn(50, 5)
        y_test = np.random.randint(0, 2, (50, 1))

        from src.ml.classifier import train_mlp

        history = train_mlp(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_hidden=5,
            epochs=1000,
            learning_rate=0.1,
            beta=0.9,
        )

        # Loss should decrease over time
        assert history["train_cost"][-1] < history["train_cost"][0]


class TestSplitIntoBatches:
    """Test sequence batch splitting."""

    def test_batch_output_shape(self):
        """Test that batching produces correct shapes."""
        seq1 = [0] * 100
        seq2 = [1] * 100
        X, y = split_into_batches(seq1, seq2, batch_size=10)

        # Each sequence produces (100 - 10) = 90 batches
        assert X.shape[0] == 180  # 90 + 90
        assert X.shape[1] == 10
        assert len(y) == 180
        assert y[:90].sum() == 0  # First half is class 0
        assert y[90:].sum() == 90  # Second half is class 1

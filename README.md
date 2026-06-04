# 🎓 Unsupervised Learning Project

> **Author:** Ibrahim COULIBALY — M2 Data Science, Université Paris Saclay

## 📖 Overview

This project implements various unsupervised learning concepts including:

- **PageRank Algorithm** — Graph simulation and stationary probability computation
- **Hidden Markov Models (HMM)** — Baum-Welch estimation and Viterbi decoding
- **Stochastic Block Models (SBM)** — Web community simulation
- **Deep Learning Classifiers** — MLP with gradient descent and momentum

## 🏗️ Architecture

```
├── pages/                  # Streamlit application pages
│   ├── home.py             # Main entry point
│   ├── 01_PageRank.py      # PageRank simulation
│   ├── 02_HMM_Web.py       # HMM web communication
│   ├── 03_SBM_Community.py # SBM community simulation
│   └── 04_Classifier.py    # MLP classifier training
├── src/
│   ├── simulations/        # Simulation modules
│   │   └── pagerank.py
│   ├── models/             # ML model implementations
│   │   └── markov.py
│   ├── ml/                 # Machine learning utilities
│   │   └── classifier.py
│   └── utils/              # Configuration and constants
│       └── config.py
├── tests/                  # Unit and integration tests
│   ├── test_pagerank.py
│   ├── test_markov.py
│   └── test_classifier.py
├── .github/workflows/      # CI/CD pipeline
│   └── ci.yml
├── Makefile                # Build automation
├── pyproject.toml          # Project configuration
├── requirements.txt        # Production dependencies
└── requirements-dev.txt    # Development dependencies
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Production only
make install

# Or with dev tools (recommended)
make install-dev
```

### 2. Run the app

```bash
make run
```

The Streamlit app will open at `http://localhost:8501`.

### 3. Run tests

```bash
make test
```

### 4. Lint & Format

```bash
make lint    # Check code quality
make format  # Auto-fix formatting
```

## 🧪 Testing

| Module | Tests | Coverage |
|--------|-------|----------|
| PageRank Simulator | 9 | 94% |
| ML Classifier | 13 | 100% |
| Markov Models | 10 | 76% |
| **Total** | **32** | **87%** |

## 🔄 CI/CD Pipeline

This project uses GitHub Actions for automated testing:

- ✅ **Linting** — Ruff + Black checks
- ✅ **Testing** — Pytest with coverage (≥80% required)
- ✅ **Multi-version** — Python 3.10, 3.11, 3.12
- ✅ **Streamlit verification** — App loads successfully

## 📝 Features

### PageRank Algorithm
- Random directed graph generation (G(n, p) model)
- Transition matrix computation with ε-smoothing
- Stationary probability via matrix power iteration
- Markov chain simulation and convergence analysis

### Hidden Markov Models
- HMM simulation with configurable parameters
- Baum-Welch algorithm for parameter estimation
- Viterbi algorithm for state sequence decoding
- Web browsing behavior analysis

### Stochastic Block Models
- Community structure simulation
- Random walk on community graphs
- Transition matrix analysis (A1 vs A2)

### Deep Learning
- Multilayer Perceptron from scratch
- Gradient descent with momentum
- Binary classification with BCE loss

## 🔧 Configuration

All default parameters are centralized in `src/utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_NUM_NODES` | 8 | Graph size |
| `DEFAULT_EDGE_PROBABILITY` | 0.5 | Edge creation probability |
| `DEFAULT_EPSILON` | 0.05 | Smoothing parameter |
| `DEFAULT_EPOCHS` | 5000 | Training epochs |
| `DEFAULT_LEARNING_RATE` | 0.1 | MLP learning rate |

## 📄 License

This project was created as part of an academic coursework.

---

**Streamlit App:** https://coulibaly-b-unsupervised-learning-unsupervised-xu88cz.streamlit.app

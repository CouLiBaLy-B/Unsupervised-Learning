"""Constants and configuration for the unsupervised learning project."""

# Default social network names for PageRank simulation
DEFAULT_SOCIAL_NETWORKS = [
    "Facebook",
    "Whatsapp",
    "Instagram",
    "Twitter",
    "Teams",
    "Discord",
    "LinkedIn",
    "Snap",
    "Bloomberg",
    "Slack",
    "Github",
]

# Default domain names for Hidden Markov Model
DEFAULT_DOMAINS = ["Sport", "Culture", "Beauty"]

# Default keywords for HMM emission
DEFAULT_KEYWORDS = [
    "Abdominaux",
    "Cosmétiques",
    "Livres",
    "Age",
    "Force",
    "Endurance",
    "Résilience",
    "Crème",
    "Histoire",
    "Mathématiques",
]

# Default transition matrix for HMM (3x3)
DEFAULT_TRANSITION_MATRIX = [
    [0.7, 0.2, 0.1],
    [0.25, 0.7, 0.05],
    [0.1, 0.1, 0.8],
]

# Default emission matrix for HMM (3x10)
DEFAULT_EMISSION_MATRIX = [
    [0.2, 0.0, 0.1, 0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0],
    [0.0, 0.1, 0.3, 0.2, 0.0, 0.0, 0.1, 0.0, 0.2, 0.1],
    [0.1, 0.3, 0.0, 0.2, 0.1, 0.0, 0.1, 0.2, 0.0, 0.0],
]

# PageRank simulation defaults
DEFAULT_NUM_NODES = 8
DEFAULT_EDGE_PROBABILITY = 0.5
DEFAULT_EPSILON = 0.05
DEFAULT_POWER_ITERATIONS = 1000

# HMM simulation defaults
DEFAULT_HMM_SIMULATION_LENGTH = 100
DEFAULT_HMM_SIMULATION_WIDTH = 30

# ML training defaults
DEFAULT_HIDDEN_NEURONS = 10
DEFAULT_EPOCHS = 5000
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MOMENTUM_BETA = 0.9
DEFAULT_TEST_SIZE = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_WEIGHT_INIT_SCALE = 0.01
DEFAULT_EPSILON_NUMERICAL = 1e-16

# SBM simulation defaults
DEFAULT_SBM_SIZE = 90
DEFAULT_SBM_ALPHA = 0.15
DEFAULT_SBM_BETA = 0.05
DEFAULT_SBM_PI = [1 / 3, 1 / 3, 1 / 3]

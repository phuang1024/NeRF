import torch

# Data parameters
# TODO

# Model hyperparameters
# Sine embeddings
FREQ_MIN = 1
FREQ_MAX = 256
EMBED_DIM = 256
# NeRF MLP model
MLP_DIM = 256
MLP_DEPTH = 8

# Training parameters
BATCH_SIZE = 64
BATCH_PER_STEP = 1
EPOCHS = 5
SAVE_INTERVAL = 5
LR_START = 1e-3
LR_END = 1e-5

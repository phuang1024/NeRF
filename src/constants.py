import torch

# Data parameters
# TODO

# Model hyperparameters
# Sine embeddings
FREQ_MIN = 0.1
FREQ_MAX = 1000
EMBED_DIM = 32
# NeRF MLP model
MLP_DIM = 256
MLP_DEPTH = 8
# Rendering
CLIPPING = 4
RENDER_STEPS = 32

# Training parameters
BATCH_SIZE = 256
BATCH_PER_STEP = 1
EPOCHS = 2
SAVE_INTERVAL = 1
LR_START = 1e-4
LR_END = 1e-5

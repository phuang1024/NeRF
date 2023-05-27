import torch

# Data parameters
# TODO

# Model hyperparameters
# Sine embeddings
FREQ_MIN = 0.1
FREQ_MAX = 1000
EMBED_DIM = 32
# NeRF MLP model
MLP_DIM = 1024
MLP_DEPTH = 8
# Rendering
CLIPPING = 4
RENDER_STEPS = 32

# Training parameters
BATCH_SIZE = 1024
BATCH_PER_STEP = 1
EPOCHS = 30
SAVE_INTERVAL = 1
LR_START = 5e-4
LR_END = 5e-5

import torch

# Data parameters
# TODO

# Model hyperparameters
# Sine embeddings
FREQ_MIN = 1
FREQ_MAX = 256
EMBED_DIM = 16
# NeRF MLP model
MLP_DIM = 256
MLP_DEPTH = 8
# Rendering
CLIPPING = 4
RENDER_STEPS = 32

# Training parameters
BATCH_SIZE = 1024
BATCH_PER_STEP = 1
EPOCHS = 5
SAVE_INTERVAL = 1
LR_START = 1e-4
LR_END = 1e-5

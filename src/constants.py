import torch

# Data parameters

# Model hyperparameters
# Sine embeddings
FREQ_MIN = 0.1
FREQ_MAX = 50
EMBED_DIM = 16
# NeRF MLP model
MLP_DIM = 256
MLP_DEPTH = 8
# Rendering
CLIP_START = 1
CLIP_END = 4
RENDER_STEPS = 32

# Training parameters
BATCH_SIZE = 1024
BATCH_PER_STEP = 1
EPOCHS = 30
SAVE_INTERVAL = 1
LR_START = 5e-4
LR_END = 5e-5

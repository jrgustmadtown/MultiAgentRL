import torch

warnings_filter_message = ".*equilibria was returned.*"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-3
GRID_SIZE = 5
GAMMA = 0.9
TARGET_UPDATE_EVERY = 500
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 32
MIN_BUFFER_SIZE = 64
GRADIENT_STEPS = 4

# Rewards normalized: scale factor 1/0.9 to get max reward = +1
# Original ratios preserved: crash:stay:living:grid = 10:5:1:5
CRASH_PENALTY = -10 / 9
STAY_PENALTY = -5 / 9
LIVING_COST = 1 / 9
GRID_REWARD_MAX = 5 / 9

ACTIONS = ["U", "D", "L", "R"]

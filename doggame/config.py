import numpy as np
import torch

warnings_filter_message = ".*equilibria was returned.*"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR = 5e-4
STEP_SIZE = 0.1  # Movement step in [0,1] space
GAMMA = 0.5
TAU = 0.001  # Polyak averaging coefficient for soft target updates
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 32
MIN_BUFFER_SIZE = 64
GRADIENT_STEPS = 4
DEFAULT_HORIZON = 10

# House positions
DEFAULT_HOUSE1 = (0.25, 0.25)
DEFAULT_HOUSE2 = (0.75, 0.75)

# 17 actions, stay + 16 directions - every 22.5 degrees
# 0 degrees = East, counterclockwise
_angles_deg = [
    0,
    22.5,
    45,
    67.5,
    90,
    112.5,
    135,
    157.5,
    180,
    202.5,
    225,
    247.5,
    270,
    292.5,
    315,
    337.5,
]
ACTION_DIRS = {0: (0, 0)}
for i, deg in enumerate(_angles_deg):
    rad = np.radians(deg)
    ACTION_DIRS[i + 1] = (np.cos(rad), np.sin(rad))

ACTION_NAMES = [
    "Stay",
    "E",
    "ENE",
    "NE",
    "NNE",
    "N",
    "NNW",
    "NW",
    "WNW",
    "W",
    "WSW",
    "SW",
    "SSW",
    "S",
    "SSE",
    "SE",
    "ESE",
]
NUM_ACTIONS = 17

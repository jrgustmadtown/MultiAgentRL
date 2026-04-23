import numpy as np
import torch

warnings_filter_message = ".*equilibria was returned.*"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training and environment defaults
LR = 5e-4
GAMMA = 0.9
TAU = 0.001
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 32
MIN_BUFFER_SIZE = 64
GRADIENT_STEPS = 4
TARGET_UPDATE_EVERY = 200

DEFAULT_ITERATIONS = 8000
DEFAULT_HORIZON = 30
STEP_SIZE = 0.10
PUSH_SCALE = 1.0
SUCCESS_RADIUS = 0.05

# Reward shaping
ALPHA_GLOBAL = 1.0
BETA_AXIS = 0.5
STEP_PENALTY = -0.01
SUCCESS_BONUS = 5.0

# Radii
BOULDER_RADIUS = 0.04
PLAYER_RADIUS = 0.03
HOLE_RADIUS = 0.05

# Default spawn positions
DEFAULT_P1_START = (0.10, 0.20)
DEFAULT_P2_START = (0.30, 0.20)
DEFAULT_BOULDER_START = (0.20, 0.40)
DEFAULT_HOLE = (0.80, 0.40)

# World bounds
WORLD_MIN = 0.0
WORLD_MAX = 1.0

# 17 actions: stay + 16 directions
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
ACTION_DIRS = {0: (0.0, 0.0)}
for i, deg in enumerate(_angles_deg):
    rad = np.radians(deg)
    ACTION_DIRS[i + 1] = (float(np.cos(rad)), float(np.sin(rad)))

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
NUM_ACTIONS = len(ACTION_NAMES)
NUM_JOINT_ACTIONS = NUM_ACTIONS * NUM_ACTIONS

# Simple maze preset: list of axis-aligned rectangles (x_min, y_min, x_max, y_max)
# Step 3 will finalize and validate wall collision behavior.
DEFAULT_WALLS = [
    (0.45, 0.00, 0.55, 0.60),
]

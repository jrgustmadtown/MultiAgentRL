import os
import sys

import torch.nn as nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utilities.paths import module_output_dir


OUTPUT_DIR = module_output_dir(__file__)


def output_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


def export_weights(net, filepath, player_info=""):
    """Export network weights with bias as last entry in each row."""
    with open(filepath, "w") as f:
        layer_idx = 0

        for _, module in net.net.named_children():
            if isinstance(module, nn.Linear):
                w = module.weight.data.cpu().numpy()
                b = module.bias.data.cpu().numpy()

                if layer_idx > 0:
                    f.write("-----\n")

                for j in range(w.shape[0]):
                    row = list(w[j]) + [b[j]]
                    f.write(",".join(f"{v:.6f}" for v in row) + "\n")

                layer_idx += 1

        f.write("=====\n")
        f.write("# Layer 0: Linear(4 -> 256) - 256 rows x 5 cols (4 weights + bias)\n")
        f.write("# Layer 1: Linear(256 -> 256) - 256 rows x 257 cols (256 weights + bias)\n")
        f.write("# Layer 2: Linear(256 -> 289) - 289 rows x 257 cols (256 weights + bias)\n")
        f.write("# Format: each row is [input_weights..., bias] for one output neuron\n")
        f.write("# Architecture: 4 -> 256 -> 256 -> 289\n")
        f.write("# Activation: ReLU\n")
        if player_info:
            f.write(f"# {player_info}\n")

    print(f"Saved weights to {filepath}")

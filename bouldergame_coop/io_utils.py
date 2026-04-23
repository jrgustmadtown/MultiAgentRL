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
    """Export network weights in standard matrix format."""
    with open(filepath, "w") as f:
        layer_idx = 0
        for _, module in net.net.named_children():
            if isinstance(module, nn.Linear):
                w = module.weight.data.cpu().numpy().T
                b = module.bias.data.cpu().numpy()

                if layer_idx > 0:
                    f.write("-----\n")

                f.write(f"# Layer {layer_idx}: Linear({w.shape[0]} -> {w.shape[1]})\n")
                f.write("# Weight matrix W (W_ij = weight from unit i to unit j):\n")
                for row in w:
                    f.write(",".join(f"{v:.6f}" for v in row) + "\n")

                f.write("# Bias vector:\n")
                f.write(",".join(f"{v:.6f}" for v in b) + "\n")
                layer_idx += 1

        f.write("-----\n")
        f.write("# Metadata\n")
        f.write("# Architecture: 8 -> 256 -> 256 -> (17*17)\n")
        f.write("# Output: Joint-action Q-values\n")
        if player_info:
            f.write(f"# {player_info}\n")

    print(f"Saved weights to {filepath}")

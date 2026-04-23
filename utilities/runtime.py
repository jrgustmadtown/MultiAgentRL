import os
import sys


def ensure_repo_root_on_path(module_file):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(module_file)))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    return repo_root


def is_headless_matplotlib(plt_module):
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    backend = plt_module.get_backend().lower()
    return backend == "agg" or not has_display

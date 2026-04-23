import os


def module_output_dir(module_file):
    return os.path.dirname(os.path.abspath(module_file))


def module_output_path(module_file, filename):
    return os.path.join(module_output_dir(module_file), filename)

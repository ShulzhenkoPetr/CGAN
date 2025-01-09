from types import SimpleNamespace
import yaml
import numpy as np
import torch


def load_config(config_path: str):
    """Reads a YAML config and returns the content as a dict."""
    try:
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
            dot_dict = SimpleNamespace(**data)
            return dot_dict
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def set_seed(seed: int = 42):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
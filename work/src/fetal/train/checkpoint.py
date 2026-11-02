import os
import torch


def atomic_save(obj, path):
    """Save a PyTorch object atomically by writing to a temporary file then swapping."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


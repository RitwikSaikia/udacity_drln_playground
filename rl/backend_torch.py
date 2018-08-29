import torch

_DEVICE = None


def set_device(device):
    global _DEVICE
    _DEVICE = device


def _device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _DEVICE

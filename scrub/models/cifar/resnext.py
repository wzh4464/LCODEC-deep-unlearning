import torch


def resnext(**kwargs):
    """Constructs a ResNeXt."""
    return torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=False)

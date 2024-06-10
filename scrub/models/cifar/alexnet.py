import torch


def alexnet(**kwargs):
    return torch.hub.load(
        "pytorch/vision:v0.10.0", "alexnet", pretrained=False, num_classes=10
    )

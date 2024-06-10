import torch


def densenet(**kwargs):
    return torch.hub.load(
        "pytorch/vision:v0.10.0",
        "densenet121",
        pretrained=False,
        num_classes=10,
    )

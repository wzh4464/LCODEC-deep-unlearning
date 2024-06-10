import torch


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return torch.hub.load(
        "pytorch/vision:v0.10.0", "vgg11", pretrained=False, num_classes=10
    )


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return torch.hub.load(
        "pytorch/vision:v0.10.0", "vgg11_bn", pretrained=False, num_classes=10
    )


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return VGG(make_layers(cfg["B"]), **kwargs)


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg["B"], batch_norm=True), **kwargs)


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return VGG(make_layers(cfg["D"]), **kwargs)


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg["D"], batch_norm=True), **kwargs)


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return VGG(make_layers(cfg["E"]), **kwargs)


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg["E"], batch_norm=True), **kwargs)

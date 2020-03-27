import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish


def _efficientnet(model_name, pretrained):
    if pretrained:
        model = EfficientNet.from_pretrained(model_name, advprop=True)
    else:
        model = EfficientNet.from_name(model_name)
    
    return model


def efficientnet_b0(model_name='efficientnet-b0', pretrained=False):
    return _efficientnet(model_name, pretrained)


def efficientnet_b1(model_name='efficientnet-b1', pretrained=False):
    return _efficientnet(model_name, pretrained)


def efficientnet_b2(model_name='efficientnet-b2', pretrained=False):
    return _efficientnet(model_name, pretrained)


def efficientnet_b3(model_name='efficientnet-b3', pretrained=False):
    return _efficientnet(model_name, pretrained)


def efficientnet_b4(model_name='efficientnet-b4', pretrained=False):
    return _efficientnet(model_name, pretrained)


def efficientnet_b5(model_name='efficientnet-b5', pretrained=False):
    return _efficientnet(model_name, pretrained)


def efficientnet_b6(model_name='efficientnet-b6', pretrained=False):
    return _efficientnet(model_name, pretrained)


def efficientnet_b7(model_name='efficientnet-b7', pretrained=False):
    return _efficientnet(model_name, pretrained)

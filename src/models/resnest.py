import torch


def resnest50(pretrained=True):
    return torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrained)

def resnest101(pretrained=True):
    return torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=pretrained)

def resnest200(pretrained=True):
    return torch.hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=pretrained)

def resnest269(pretrained=True):
    return torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=pretrained)



import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import layer
from models import densenet, efficientnet, mobilenet, resnet, senet
from dataset.custom_dataset import CustomDataset

model_encoder = {
    # densenet
    'densenet121': densenet.densenet121,

    # efficientnet
    'efficientnet_b0': efficientnet.efficientnet_b0,
    'efficientnet_b1': efficientnet.efficientnet_b1,
    'efficientnet_b2': efficientnet.efficientnet_b2,
    'efficientnet_b3': efficientnet.efficientnet_b3,
    'efficientnet_b4': efficientnet.efficientnet_b4,
    'efficientnet_b5': efficientnet.efficientnet_b5,
    'efficientnet_b6': efficientnet.efficientnet_b6,
    'efficientnet_b7': efficientnet.efficientnet_b7,

    # mobilenet
    'mobilenet': mobilenet.mobilenet_v2,

    # resnet
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
    'resnet152': resnet.resnet152,
    'resnext50_32x4d': resnet.resnext50_32x4d,
    'resnext101_32x8d': resnet.resnext101_32x8d,
    'wide_resnet50_2': resnet.wide_resnet50_2,
    'wide_resnet101_2': resnet.wide_resnet101_2,

    # senet
    'se_resnext50_32x4d': senet.se_resnext50_32x4d,
    'se_resnext101_32x4d': senet.se_resnext101_32x4d,
}


def set_channels(child, cfg):
    if cfg.model.n_channels < 3:
        child_weight = child.weight.data[:, :cfg.model.n_channels, :, :]
    else:
        child_weight = torch.cat([child.weight.data[:, :, :, :], child.weight.data[:, :int(cfg.model.n_channels - 3), :, :]], dim=1)
    setattr(child, 'in_channels', cfg.model.n_channels)

    if cfg.model.pretrained:
        setattr(child.weight, 'data', child_weight)


def replace_channels(model, cfg):
    if cfg.model.name.startswith('densenet'):
        set_channels(model.features[0], cfg)
    elif cfg.model.name.startswith('efficientnet'):
        set_channels(model._conv_stem, cfg)
    elif cfg.model.name.startswith('mobilenet'):
        set_channels(model.features[0][0], cfg)
    elif cfg.model.name.startswith('se_resnext'):
        set_channels(model.layer0.conv1, cfg)
    elif cfg.model.name.startswith('resnet') or cfg.model.name.startswith('resnex') or cfg.model.name.startswith('wide_resnet'):
        set_channels(model.conv1, cfg)


def replace_fc(model, cfg):
    if cfg.model.metric:
        classes = 1000
    else:
        classes = cfg.model.n_classes

    if cfg.model.name.startswith('densenet'):
        fc_input = getattr(model.classifier, 'in_features')
        model.classifier = nn.Linear(fc_input, classes)
    elif cfg.model.name.startswith('efficientnet'):
        fc_input = getattr(model._fc, 'in_features')
        model._fc = nn.Linear(fc_input, classes)
    elif cfg.model.name.startswith('mobilenet'):
        fc_input = getattr(model.classifier[1], 'in_features')
        model.classifier[1] = nn.Linear(fc_input, classes)
    elif cfg.model.name.startswith('se_resnext'):
        fc_input = getattr(model.last_linear, 'in_features')
        model.last_linear = nn.Linear(fc_input, classes)
    elif cfg.model.name.startswith('resnet') or cfg.model.name.startswith('resnex') or cfg.model.name.startswith('wide_resnet'):
        fc_input = getattr(model.fc, 'in_features')
        model.fc = nn.Linear(fc_input, classes)
    return model


def replace_pool(model, cfg):
    avgpool = getattr(layer, cfg.model.avgpool.name)(**cfg.model.avgpool.params)
    if cfg.model.name.startswith('efficientnet'):
        model._avg_pooling = avgpool
    elif cfg.model.name.startswith('se_resnext'):
        model.avg_pool = avgpool
    elif cfg.model.name.startswith('resnet') or cfg.model.name.startswith('resnex') or cfg.model.name.startswith('wide_resnet'):
        model.avgpool = avgpool
    return model


def get_model(cfg):
    model = model_encoder[cfg.model.name](pretrained=cfg.model.pretrained)
    if cfg.model.n_channels != 3:
        replace_channels(model, cfg)
    model = replace_fc(model, cfg)
    if cfg.model.avgpool:
        model = replace_pool(model, cfg)
    return model


def get_loss(cfg):
    loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    return loss


def get_dataloader(df, labels, cfg):
    dataset = CustomDataset(df, labels, cfg)
    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optimizer.name)(params=parameters, **cfg.optimizer.params)
    return optim


def get_scheduler(cfg, optimizer):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **cfg.scheduler.params,
        )
    else:
        scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.name)(
            optimizer,
            **cfg.scheduler.params,
        )
    return scheduler
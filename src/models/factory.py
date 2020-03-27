import densenet
import efficientnet
import mobilenet
import resnet
import senet


model_encoder = {
    # densenet
    'densenet121': densenet.densenet121,

    # efficientnet
    'efficientnet-b0': efficientnet.efficientnet_b0,
    'efficientnet-b1': efficientnet.efficientnet_b1,
    'efficientnet-b2': efficientnet.efficientnet_b2,
    'efficientnet-b3': efficientnet.efficientnet_b3,
    'efficientnet-b4': efficientnet.efficientnet_b4,
    'efficientnet-b5': efficientnet.efficientnet_b5,
    'efficientnet-b6': efficientnet.efficientnet_b6,
    'efficientnet-b7': efficientnet.efficientnet_b7,

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
    'wide_resnet50_2': renset.wide_resnet50_2,
    'wide_resnet101_2': resnet.wide_resnet101_2,

    # senet
    'se_resnext50_32x4d': senet.se_resnext50_32x4d,
    'se_resnext101_32x4d': senet.se_resnext101_32x4d,

}

def set_in_channel(child, n_channels, pretrained):
    if n_channels < 3:
        child_weight = child.weight.data[:, :n_channels, :, :]
    else:
        child_weight = torch.cat([child.weight.data[:, :, :, :], child.weight.data[:, :int(n_channels - 3), :, :])
    setattr(child, 'in_channels', n_channels)

    if pretrained:
        setattr(child.weight, 'data', child_weight)


def replace_in_channels(model, model_name, n_channels):
    if model_name in ['densenet121']:
        set_in_channel(model.features[0], n_channels, pretrained)
    elif model_name in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 
                        'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']:
        set_in_channel(model._conv_stem, n_channels, pretrained)
    elif model_name in ['mobilenet']:
        set_in_channel(model.features[0][0], n_channels, pretrained)
    elif model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                        'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']:
        set_in_channel(model.conv1, n_channels, pretrained)
    elif model_name in ['se_resnext50_32x4d', 'se_resnext101_32x4d']:
        set_in_channel(model.layer0.conv1, n_channels, pretrained)


def make_model(model_name, n_channels=3, pretrained=Fales):
    model = model_encoder[model_name]

    if n_channels != 3:
        replace_in_channels(model, model_name, n_channels, pretrained)

    return model

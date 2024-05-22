import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision import models
import pretrainedmodels
from .efficientnet import efficientnet_b0
from .efficientnet import efficientnet_b1
from .efficientnet import efficientnet_b2
from .efficientnet import efficientnet_b3
from .efficientnet import efficientnet_b4
from .efficientnet import efficientnet_b5
from .efficientnet import efficientnet_b6
from .efficientnet import efficientnet_b7


class FCNorm(nn.Module):
    def __init__(self, in_features, out_features):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out


def get_model(model_name, pretrained=False):
    """Returns a CNN model
    Args:
      model_name: model name
      pretrained: True or False
    Returns:
      model: the desired model
    Raises:
      ValueError: If model name is not recognized.
    """
    if pretrained == False:
        pt = None
    else:
        pt = 'imagenet'
    print(f"model pretrained: {pretrained}, mode: {pt}", )

    if model_name == 'Vgg11':
        return models.vgg11(pretrained=pretrained)
    elif model_name == 'Vgg13':
        return models.vgg13(pretrained=pretrained)
    elif model_name == 'Vgg16':
        return models.vgg16(pretrained=pretrained)
    elif model_name == 'Vgg19':
        return models.vgg19(pretrained=pretrained)
    elif model_name == 'Resnet18':
        return models.resnet18(pretrained=pretrained)
    elif model_name == 'Resnet34':
        return models.resnet34(pretrained=pretrained)
    elif model_name == 'Resnet50':
        return models.resnet50(pretrained=pretrained)
    elif model_name == 'Resnet101':
        return models.resnet101(pretrained=pretrained)
    elif model_name == 'Resnet152':
        return models.resnet152(pretrained=pretrained)
    elif model_name == 'Dense121':
        return models.densenet121(pretrained=pretrained)
    elif model_name == 'Dense169':
        return models.densenet169(pretrained=pretrained)
    elif model_name == 'Dense201':
        return models.densenet201(pretrained=pretrained)
    elif model_name == 'Dense161':
        return models.densenet161(pretrained=pretrained)
    elif model_name == 'SENet50':
        return pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained=pt)
    elif model_name == 'SENet101':
        return pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained=pt)
    elif model_name == 'SENet152':
        return pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained=pt)
    elif model_name == 'SENet154':
        return pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b0':
        return efficientnet_b0(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b1':
        return efficientnet_b1(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b2':
        return efficientnet_b2(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b3':
        return efficientnet_b3(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b4':
        return efficientnet_b4(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b5':
        return efficientnet_b5(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b6':
        return efficientnet_b6(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b7':
        return efficientnet_b7(num_classes=1000, pretrained=pt)
    else:
        raise ValueError('Name of model unknown %s' % model_name)


def modify_last_layer(model_name, model, num_classes, normed=False, bias=True):
    """modify the last layer of the CNN model to fit the num_classes
    Args:
      model_name: model name
      model: CNN model
      num_classes: class number
    Returns:
      model: the desired model
    """

    if 'Vgg' in model_name:
        num_ftrs = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = classifier(num_ftrs, num_classes, normed, bias)
        last_layer = model.classifier._modules['6']
    elif 'Dense' in model_name:
        num_ftrs = model.classifier.in_features
        model.classifier = classifier(num_ftrs, num_classes, normed, bias)
        last_layer = model.classifier
    elif 'Resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = classifier(num_ftrs, num_classes, normed, bias)
        last_layer = model.fc
    elif 'Efficient' in model_name:
        num_ftrs = model._fc.in_features
        model._fc = classifier(num_ftrs, num_classes, normed, bias)
        last_layer = model._fc
    else:
        num_ftrs = model.last_linear.in_features
        model.last_linear = classifier(num_ftrs, num_classes, normed, bias)
        last_layer = model.last_linear
    # print(model)
    return model, last_layer


def classifier(num_features, num_classes, normed, bias):
    if normed:
        last_linear = FCNorm(num_features, num_classes)
    else:
        last_linear = nn.Linear(num_features, num_classes, bias=bias)
    return last_linear


def get_feature_length(model_name, model):
    """get the feature length of the last feature layer
    Args:
      model_name: model name
      model: CNN model
    Returns:
      num_ftrs: the feature length of the last feature layer
    """
    if 'Vgg' in model_name:
        num_ftrs = model.classifier._modules['6'].in_features
    elif 'Dense' in model_name:
        num_ftrs = model.classifier.in_features
    elif 'Resnet' in model_name:
        num_ftrs = model.fc.in_features
    elif 'Efficient' in model_name:
        num_ftrs = model._fc.in_features
    elif 'RegNet' in model_name:
        num_ftrs = model.head.fc.in_features
    else:
        num_ftrs = model.last_linear.in_features

    return num_ftrs
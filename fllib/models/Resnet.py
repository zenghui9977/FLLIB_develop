import torchvision
import torch.nn as nn


def ResNet18(num_classes=10, channels=3):
    model = torchvision.models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def ResNet34(num_classes=10, channels=3):
    model = torchvision.models.resnet34(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def ResNet50(num_classes=10, channels=3):
    model = torchvision.models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model



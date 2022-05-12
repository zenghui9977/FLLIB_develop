
import torchvision
import torch.nn as nn


def MobileNetV2(num_classes=10, channels=3):
    model = torchvision.models.mobilenet_v2(pretrained=True)   
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)

    return model

def MobileNetV3_large(num_classes=10, channels=3):
    model = torchvision.models.mobilenet_v3_large(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)

    return model    


def MobileNetV3_small(num_classes=10, channels=3):
    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)
    
    return model  
 
MobileNetV3_large()



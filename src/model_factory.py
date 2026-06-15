import torch.nn as nn
from torchvision import models


def create_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError("Choose one of: resnet18, resnet34, resnet50")

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 9)

    return model
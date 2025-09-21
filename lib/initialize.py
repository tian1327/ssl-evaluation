import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from lib.resnet import resnet50, resnet101

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False,
                    use_pretrained=True, logger=None):
    model_ft = None

    if model_name == "resnet101":
        """ Resnet101
        """
        # model_ft = models.resnet101(pretrained=use_pretrained)
        model_ft = resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        # model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        print(f"model.fc.in_features: {model_ft.fc.in_features}")
        print(f"model.fc.out_features: {model_ft.fc.out_features}")        

    elif model_name.split("_")[0] == 'dinov2':
        """ dinov2
        """
        model_ft = torch.hub.load('facebookresearch/dinov2', model_name)
        print('Loaded model: ', model_name)
        # for name, module in model_ft.named_modules():
            # print(name)
        print(f'model_ft.linear_head.in_features: {model_ft.linear_head.in_features}')
        print(f'model_ft.linear_head.out_features: {model_ft.linear_head.out_features}')
        

        # Assuming model_ft is your loaded model
        # dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the size of the input as per your model's requirement
        # output = model_ft(dummy_input)
        # print("Output size:", output.size())

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.linear_head.in_features
        # num_ftrs = 768
        model_ft.linear_head = nn.Linear(num_ftrs, num_classes)
        print(f'model_ft.linear_head.in_features: {model_ft.linear_head.in_features}')
        print(f'model_ft.linear_head.out_features: {model_ft.linear_head.out_features}')


    return model_ft
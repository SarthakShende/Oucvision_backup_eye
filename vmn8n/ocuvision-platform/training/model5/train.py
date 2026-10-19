import torch
import timm
from pathlib import Path
import yaml

def build_model(num_classes=13):
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
    return model

def main():
    pass

if __name__ == '__main__':
    main()

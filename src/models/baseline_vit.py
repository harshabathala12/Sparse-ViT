import timm
import torch.nn as nn

def build_baseline_vit(num_classes=10):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    return model

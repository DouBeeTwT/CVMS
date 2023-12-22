import torch
import torch.nn as nn

# https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/model.py

class BiFPN(nn.Module):
    def __init__(self) -> None:
        super(BiFPN, self).__init__()

    def forward(self, x):
        pass
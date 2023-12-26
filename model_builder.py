import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

dim_list = [16, 24, 40, 112, 320]

class Model(nn.Module):
    def __init__(self, backbone:str, neck:str, head:str) -> None:
        super(Model, self).__init__()
        if backbone == "EfficientNet":
            from Model.Backbone.EfficientNet import EfficientNet
            self.backbone = EfficientNet(channel_figure=1, classes=4, dim_list=[16, 24, 40, 112, 320])
        
        if neck == "BiFPN":
            from Model.Neck.BiFPN import BiFPN_layer
            self.neck = BiFPN_layer(feature_list=[16, 24, 40, 112, 320])

        if head == "seg":
            from Model.Head.seg import seg
            self.head = seg(channel_input=16, classes=4)

    def forward(self, x:Tensor) -> Tensor:
        f_list_b, f_end = self.backbone(x)
        f_list_n = self.neck(f_list_b)
        out = self.head(f_list_n[0])
        return out
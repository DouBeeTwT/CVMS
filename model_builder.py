import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

dim_list = [16, 24, 40, 112, 320]

class Model(nn.Module):
    def __init__(self, backbone:str, neck:str, head:str, channel_figure:int=1, dim_list:list=[16, 24, 40, 112, 320],
                 classes_seg:int=2, classes_cls:int=2) -> None:
        """
        Args:
            backbone: Choose the backbone by name
                'EfficientNet | EfficientNetV2 | UnetEncoder'
            neck: Choose the neck by name
                'BiFPN | UnetDecoder'
            head: Choose the head by name
                'cls | seg | seg+cls'
        """
        super(Model, self).__init__()
        if backbone == "EfficientNet":
            from Model.Backbone.EfficientNet import EfficientNet
            self.backbone = EfficientNet(channel_figure=channel_figure, dim_list=dim_list)
        
        if neck == "BiFPN":
            from Model.Neck.BiFPN import BiFPN_layer
            self.neck = BiFPN_layer(feature_list=dim_list)

        if head == "seg":
            from Model.Head.seg import seg
            self.head = seg(channel_input=dim_list[0], classes=classes_seg)
        elif head == "cls":
            from Model.Head.cls import cls
            self.head = cls(channel_input=dim_list[-1], classes=classes_cls, dim=dim_list[-1]*4)
        elif head == "seg+cls":
            pass

    def forward(self, x:Tensor) -> Tensor:
        f_list_b = self.backbone(x)
        f_list_n = self.neck(f_list_b)
        out = self.head(f_list_n)
        return out
    
if __name__ == "__main__":
    dim_list = [16, 24, 40, 112, 320]
    model = Model(backbone="EfficientNet", neck="BiFPN", head="seg", channel_figure=1, classes_seg=4)
    input = torch.rand(2, 1, 1024, 1024)
    out = model(input)
    print("Shape of output: ", out.shape)
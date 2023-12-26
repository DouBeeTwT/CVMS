import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class UpSample(nn.Module):
    def __init__(self, feature_input:int, feature_output:int, mode:str="nearest") -> None:
        """
        Args:
            feture_input: in_channels for Conv2d()
            feture_output: out_channels for Conv2d()
            mode: algorithm used for upsampling. Default: 'nearest'
                  'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact' 
        """
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(feature_input, feature_output, kernel_size=1)
        self.mode = mode

    def forward(self, x:Tensor) -> Tensor:
        return self.layer(F.interpolate(x, scale_factor=2, mode=self.mode))



class DownSample(nn.Module):
    def __init__(self, feature_input:int, feature_output:int) -> None:
        """
        Args:
            feture_input: in_channels for Conv2d()
            feture_output: out_channels for Conv2d()
        """
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(feature_input, feature_output, kernel_size=3,
                      stride=2, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(feature_output)
        )
    
    def forward(self, x:Tensor) -> Tensor:
        return self.layer(x)

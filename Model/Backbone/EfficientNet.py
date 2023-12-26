import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from collections import OrderedDict

def adjust_repeats(repeats:int, depth_coefficient:float):
    return int(math.ceil(depth_coefficient * repeats))
    
def adjust_channel(channel:int, width_coefficient:float, channel_min:int=None ,divisor:int=8):
    if channel_min == None:
        channel_min = divisor
    channel_adjust = max(channel_min, int(channel*width_coefficient + divisor/2) // divisor * divisor)
    if channel_adjust < 0.9 * channel * width_coefficient:
        channel_adjust += divisor
    return channel_adjust

class SqueezeExcitationModule(nn.Module):
    def __init__(self, channel_input:int, channel_expand:int, squeeze_factor:int=4):
        """
        channel_input:int   |   input channels of the block
        channel_expand:int  |   expand channels of the block,
                            |   to make the same channel after deepwidth conv
        squeeze_factor:int=4|   contrl the number of nodes in the first FC layer,
                            |   number of nodes = channel_input // squeeze_factor
        """
        super(SqueezeExcitationModule, self).__init__()
        channel_squeeze = channel_input // squeeze_factor
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),        # Average pool of each channel
            nn.Conv2d(channel_expand, channel_squeeze, 1),  # Use 1x1 conv to instead of fc
            nn.SiLU(),                                      # alias Swish
            nn.Conv2d(channel_squeeze, channel_expand, 1),  # Use 1x1 conv to instead of fc
            nn.Sigmoid()                                    # Get scale score of each channel
        )

    def forward(self, x:Tensor) -> Tensor:
        scale = self.layer(x)
        return scale * x

class StochasticDepth(nn.Module):
    def __init__(self, p:float=0.5, training:bool = False):
        super(StochasticDepth, self).__init__()
        self.p = p
        self.training = training
    
    def forward(self, x:Tensor) -> Tensor:
        if self.p == 0 or not self.training:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # work with different dim tensors, not 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class MBConv(nn.Module):
    def __init__(self, kernel_size:int, channel_input:int, channel_output:int,
                 expanded_ratio:int, stride:int, drop_connect_rate:float,
                 width_coefficient:float):
        """
        kernel_size:int
        channel_input:int
        channel_output:int
        expanded_ratio:int
        stride:int
        drop_connect_rate:float
        width_coefficient:float
        """
        super(MBConv, self).__init__()
        self.kernel_size = kernel_size
        self.channel_input_adj = adjust_channel(channel_input, width_coefficient)
        self.channel_expanded = self.channel_input_adj * expanded_ratio
        self.channel_output_adj = adjust_channel(channel_output, width_coefficient)
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.use_shortcut = (stride == 1 and channel_input == channel_output)

        # 1. 1x1 Conv + BN + Swish
        if expanded_ratio == 6:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(self.channel_input_adj, self.channel_expanded, 1, bias=False), 
                nn.BatchNorm2d(self.channel_expanded),
                nn.SiLU()
            )
        
        # 2. kxk Depthwise Conv + BN + Swish
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.channel_expanded, self.channel_expanded , self.kernel_size, stride=self.stride,
                      padding=int((self.kernel_size-1)/2),groups=self.channel_expanded, bias=False),
            nn.BatchNorm2d(self.channel_expanded),
            nn.SiLU()
        )

        # 3. SE Module
        self.se = SqueezeExcitationModule(self.channel_input_adj, self.channel_expanded)

        # 4. 1x1 Conv + BN
        self.down_conv = nn.Sequential(
            nn.Conv2d(self.channel_expanded, self.channel_output_adj, 1, bias=False),
            nn.BatchNorm2d(self.channel_output_adj)
        )

        # 5. Droupout
        if self.use_shortcut:
            self.droupout = StochasticDepth(drop_connect_rate)
    
    def forward(self, x:Tensor) -> Tensor:
        x_0 = x
        # 1. 1x1 Conv + BN + Swish
        if self.channel_expanded > self.channel_input_adj:
            x = self.expand_conv(x)
        # 2. kxk Depthwise Conv + BN + Swish
        x = self.depthwise_conv(x)
        # 3. SE Module
        x = self.se(x)
        # 4. 1x1 Conv + BN
        x = self.down_conv(x)
        # 5. Droupout
        if self.use_shortcut:
            x = self.droupout(x)
            x += x_0
        return x

class EfficientNet(nn.Module):
    def __init__(self, channel_figure:int, dim_list:list=[16, 24, 40, 112, 320],
                 width_coefficient:float=1.0, depth_coefficient:float=1.0,
                 drop_connect_rate:float=0.2):
        super(EfficientNet, self).__init__()
        # Parameters of stage2~stage8
        # [0:kernal size, 1:channel_input, 2:channel_output, 3:expanded_ratio, 4:stride, 5:repeats]
        self.param = {"stage2": [3, dim_list[0]*2, dim_list[0]  , 1, 1, 1],
                      "stage3": [3, dim_list[0]  , dim_list[1]  , 6, 2, 2],
                      "stage4": [5, dim_list[1]  , dim_list[2]  , 6, 2, 2],
                      "stage5": [3, dim_list[2]  , dim_list[2]*2, 6, 2, 3],
                      "stage6": [5, dim_list[2]*2, dim_list[3]  , 6, 1, 3],
                      "stage7": [5, dim_list[3]  , dim_list[3]*2, 6, 2, 4],
                      "stage8": [3, dim_list[3]*2, dim_list[4]  , 6, 1, 1]}
        self.num_MBConvs = 0
        for item in self.param.values():
            self.num_MBConvs += float(adjust_repeats(item[-1], depth_coefficient))
        self.recent_block_number = 0
        self.drop_connect_rate = drop_connect_rate
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.channel_figure = channel_figure

        # stage1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_figure,
                      out_channels=adjust_channel(dim_list[0]*2, width_coefficient),
                      kernel_size=3, padding=1, stride=1, bias=False), # stride=2 in origianl EfficentNet
            nn.BatchNorm2d(adjust_channel(dim_list[0]*2, width_coefficient))
        )

        # stage2
        self.stage2 = self.build_stage(stage_name="stage2")

        # stage3
        self.stage3 = self.build_stage(stage_name="stage3")

        # stage4
        self.stage4 = self.build_stage(stage_name="stage4")

        # stage5
        self.stage5 = self.build_stage(stage_name="stage5")

        # stage6
        self.stage6 = self.build_stage(stage_name="stage6")

        # stage7
        self.stage7 = self.build_stage(stage_name="stage7")

        # stage8
        self.stage8 = self.build_stage(stage_name="stage8")
       
    def build_stage(self, stage_name:str) -> nn.Module:
        layers = []
        param_list = self.param[stage_name]
        for index in range(adjust_repeats(param_list[5], self.depth_coefficient)):
            if index > 0:
                param_list[4] = 1 # stride = 1
                param_list[1] = param_list[2] # channel_input = channel_output
            layers.append(MBConv(kernel_size=param_list[0],
                                 channel_input=param_list[1],
                                 channel_output=param_list[2],
                                 expanded_ratio=param_list[3],
                                 stride=param_list[4],
                                 drop_connect_rate=self.drop_connect_rate*self.recent_block_number/self.num_MBConvs,
                                 width_coefficient=self.width_coefficient))
            self.recent_block_number += 1
        return nn.Sequential(*layers)
    
    def forward(self, x:Tensor) -> list:
        r0 = self.stage1(x)         # Shape r0: [b,   32,  p/1, 1024]
        r1 = self.stage2(r0)        # Shape r1: [b,   16,  p/1, 1024] *
        r2 = self.stage3(r1)        # Shape r2: [b,   24,  p/2,  512] *
        r3 = self.stage4(r2)        # Shape r3: [b,   40,  p/4,  256] *
        r4 = self.stage5(r3)        # Shape r4: [b,   80,  p/8,  128]
        r5 = self.stage6(r4)        # Shape r5: [b,  112,  p/8   128] *
        r6 = self.stage7(r5)        # Shape r6: [b,  192,  p/16,  64]
        r7 = self.stage8(r6)        # Shape r7: [b,  320,  p/16,  64] *

        return [r1, r2, r3, r5, r7]
    
if __name__ == "__main__":
    from torchsummary import summary
    model = EfficientNet(channel_figure=1, classes=4)
    summary(model, input_size=(1, 1024, 1024), device="cpu")
import torch
import torch.nn as nn
from torch import Tensor

class seg(nn.Module):
    def __init__(self, channel_input:int, classes:int) -> None:
        super(seg, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel_input, classes, kernel_size=3 ,stride=1 ,padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x:list) -> Tensor:
        return self.layer(x[0])
    
if __name__ == "__main__":
    tensor_test_input = [torch.rand(2, 16, 1024, 1024), 0, 0, 0, 0]
    model = seg(channel_input=16, classes=4)
    out = model(tensor_test_input)
    print("Shape of output:", out.shape)

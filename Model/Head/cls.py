import torch
import torch.nn as nn
from torch import Tensor

class cls(nn.Module):
    def __init__(self, channel_input:int, classes:int, dim:int) -> None:
        super(cls, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=channel_input, out_channels=dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(dim, classes),
            nn.Softmax()
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.layer(x)
    
if __name__ == "__main__":
    tensor_test_input = torch.rand(2, 320, 64, 64)
    model = cls(channel_input=320, classes=4, dim=1280)
    out = model(tensor_test_input)
    print("Shape of output:", out.shape)
    print(out)

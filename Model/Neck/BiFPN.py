import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from os.path import split, abspath
import sys
sys.path.append(split(abspath(__file__))[0].rsplit('/', 1)[0])
from ModuleBlock import UpSample, DownSample



class BiFPN_layer(nn.Module):
    def __init__(self, feature_list) -> None:
        super(BiFPN_layer, self).__init__()
        # Up Sample Layer
        self.P7i_P6m = UpSample(feature_list[4], feature_list[3])
        self.P6m_P5m = UpSample(feature_list[3], feature_list[2])
        self.P5m_P4m = UpSample(feature_list[2], feature_list[1])
        self.P4m_P3o = UpSample(feature_list[1], feature_list[0])
        # Down Sample Layer
        self.P3o_P4o = DownSample(feature_list[0], feature_list[1])
        self.P4o_P5o = DownSample(feature_list[1], feature_list[2])
        self.P5o_P6o = DownSample(feature_list[2], feature_list[3])
        self.P6o_P7o = DownSample(feature_list[3], feature_list[4])
        # Swish
        self.swish = nn.SiLU()

    def forward(self, P3i:Tensor, P4i:Tensor, P5i:Tensor, P6i:Tensor, P7i:Tensor) -> Tensor:
        # Up layer
        P6m = self.swish(P6i + self.P7i_P6m(P7i))
        P5m = self.swish(P5i + self.P6m_P5m(P6m))
        P4m = self.swish(P4i + self.P5m_P4m(P5m))
        P3o = self.swish(P3i + self.P4m_P3o(P4m))
        # Down layer
        P4o = self.swish(P4i + P4m + self.P3o_P4o(P3o))
        P5o = self.swish(P5i + P5m + self.P4o_P5o(P4o))
        P6o = self.swish(P6i + P6m + self.P5o_P6o(P5o))
        P7o = self.swish(P7i + self.P6o_P7o(P6o))
        
        return P3o, P4o, P5o, P6o, P7o



if __name__ == "__main__":
    tensor3i = torch.rand(2,  16, 1024, 1024)
    tensor4i = torch.rand(2,  24,  512,  512)
    tensor5i = torch.rand(2,  40,  256,  256)
    tensor6i = torch.rand(2, 112,  128,  128)
    tensor7i = torch.rand(2, 320,   64,   64)

    model = BiFPN_layer(feature_list=[16, 24, 40, 112, 320])
    P3o, P4o, P5o, P6o, P7o = model(tensor3i,tensor4i,tensor5i,tensor6i,tensor7i)

    print("Shape of P3o:", P3o.shape)
    print("Shape of P4o:", P4o.shape)
    print("Shape of P5o:", P5o.shape)
    print("Shape of P6o:", P6o.shape)
    print("Shape of P7o:", P7o.shape)

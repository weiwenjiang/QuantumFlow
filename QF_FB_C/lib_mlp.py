import torch
import torch.nn as nn
import math
from QF_Net.lib_util import *
from torch.nn.parameter import Parameter

class MLP(nn.Linear):
    def forward(self, input):
        binarize = BinarizeF.apply
        binary_weight = binarize(self.weight)
        if self.bias is None:
            output = F.linear(input, binary_weight)
            output = torch.div(output, input.shape[-1])
            return output
        else:
            print("Not Implement")
            sys.exit(0)
    def reset_parameters(self):
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        self.weight.lr_scale = 1. / stdv


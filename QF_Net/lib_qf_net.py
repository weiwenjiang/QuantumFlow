import torch.nn as nn
from .lib_util import *
from QF_FB_C.lib_qf_fb import *
from QF_FB_C.lib_mlp import *
import torch

## Define the NN architecture
class Net(nn.Module):
    def __init__(self,img_size,layers,with_norm,given_ang,train_ang,training,binary,classic):
        super(Net, self).__init__()

        self.in_size = img_size*img_size
        self.training = training
        self.with_norm = with_norm
        self.layer = len(layers)
        self.binary = binary
        self.classic = classic
        cur_input_size = self.in_size        
        for idx in range(self.layer):
            fc_name = "fc"+str(idx)
            if classic:
                setattr(self, fc_name, MLP(cur_input_size, layers[idx], bias=False))
            else:
                setattr(self, fc_name, QF_FB_NC(cur_input_size, layers[idx], bias=False))
            cur_input_size = layers[idx]

        if self.with_norm:
            for idx in range(self.layer):
                IAdj_name = "IAdj"+str(idx)
                BAdj_name = "BAdj"+str(idx)
                setattr(self, IAdj_name, QF_FB_BN_IAdj(num_features=layers[idx], init_ang_inc=given_ang[idx], training=train_ang))
                setattr(self, BAdj_name, QF_FB_BN_BAdj(num_features=layers[idx]))
            for idx in range(self.layer):
                bn_name = "bn"+str(idx)
                setattr(self, bn_name,nn.BatchNorm1d(num_features=layers[idx]))


    def forward(self, x, training=1):
        binarize = BinarizeF.apply
        clipfunc = ClipF.apply
        x = x.view(-1, self.in_size)
        if self.classic == 1 and self.with_norm==0:
            for layer_idx in range(self.layer):
                if self.binary and layer_idx==0:
                    x = binarize(x-0.5)
                x = getattr(self, "fc" + str(layer_idx))(x)
                x = x.pow(2)
        elif self.classic == 1 and self.with_norm==1:
            for layer_idx in range(self.layer):
                if self.binary and layer_idx==0:
                    x = (binarize(x - 0.5) + 1) / 2
                x = getattr(self, "fc" + str(layer_idx))(x)
                x = x.pow(2)
                x = getattr(self, "bn" + str(layer_idx))(x)
                x = clipfunc(x)

        elif self.classic == 0 and self.with_norm==0:
            for layer_idx in range(self.layer):
                if self.binary and layer_idx==0:
                    x = (binarize(x - 0.5) + 1) / 2
                x = getattr(self, "fc" + str(layer_idx))(x)
        else:   # Quantum Training
            if self.training == 1:
                for layer_idx in range(self.layer):
                    if self.binary and layer_idx==0:
                        x = (binarize(x-0.5)+1)/2
                    x = getattr(self, "fc"+str(layer_idx))(x)
                    x = getattr(self, "BAdj"+str(layer_idx))(x)
                    x = getattr(self, "IAdj"+str(layer_idx))(x)
            else:
                for layer_idx in range(self.layer):
                    if self.binary and layer_idx==0:
                        x = (binarize(x-0.5)+1)/2
                    x = getattr(self, "fc"+str(layer_idx))(x)
                    x = getattr(self, "BAdj"+str(layer_idx))(x, training=False)
                    x = getattr(self, "IAdj"+str(layer_idx))(x, training=False)
        return x



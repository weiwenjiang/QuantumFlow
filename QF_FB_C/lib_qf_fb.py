import torch
import torch.nn as nn
import math
from QF_Net.lib_util import *
from torch.nn.parameter import Parameter


class QF_FB_NC(nn.Linear):
    def sim_neural_comp(self, input_ori, w_ori):
        p = input_ori
        d = 4 * p * (1 - p)
        e = (2 * p - 1)
        w = w_ori
        sum_of_sq = (d + e.pow(2)).sum(-1)
        sum_of_sq = sum_of_sq.unsqueeze(-1)
        sum_of_sq = sum_of_sq.expand(p.shape[0], w.shape[0])
        diag_p = torch.diag_embed(e)
        p_w = torch.matmul(w, diag_p)
        z_p_w = torch.zeros_like(p_w)
        shft_p_w = torch.cat((p_w, z_p_w), -1)
        sum_of_cross = torch.zeros_like(p_w)
        length = p.shape[1]
        for shft in range(1, length):
            sum_of_cross += shft_p_w[:, :, 0:length] * shft_p_w[:, :, shft:length + shft]
        sum_of_cross = sum_of_cross.sum(-1)
        return (sum_of_sq + 2 * sum_of_cross) / (length ** 2)

    def forward(self, input):
        binarize = BinarizeF.apply
        binary_weight = binarize(self.weight)
        if self.bias is None:
            return self.sim_neural_comp(input, binary_weight)
        else:
            print("Bias is not supported at current version")
            sys.exit(0)

    def reset_parameters(self):
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        self.weight.lr_scale = 1. / stdv



class QF_FB_BN_IAdj(nn.Module):
    def __init__(self, num_features, init_ang_inc=1, momentum=0.1,training = False):
        super(QF_FB_BN_IAdj, self).__init__()

        self.x_running_rot = Parameter(torch.zeros((num_features)), requires_grad=False)
        self.ang_inc = Parameter(torch.tensor(init_ang_inc,dtype=torch.float32),requires_grad=True)
        self.momentum = momentum

        self.printed = False
        self.x_mean_ancle = 0
        self.x_mean_rote = 0
        self.input = 0
        self.output = 0

    def forward(self, x, training=True):
        if not training:
            if not self.printed:
                self.printed = True
            x_1 = (self.x_running_rot * x)

        else:
            self.printed = False
            x = x.transpose(0, 1)
            x_sum = x.sum(-1).unsqueeze(-1).expand(x.shape)
            x_lack_sum = x_sum + x
            x_mean = x_lack_sum / x.shape[-1]
            ang_inc = (self.ang_inc > 0).float() * self.ang_inc + 1
            y = 0.5 / x_mean
            y = y.transpose(0, 1)
            y = y / ang_inc
            y = y.transpose(0, 1)
            x_moving_rot = (y.sum(-1) / x.shape[-1])
            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot
            x_1 = y * x
            x_1 = x_1.transpose(0, 1)
        return x_1

    def reset_parameters(self):
        self.reset_running_stats()
        self.ang_inc.data.zeros_()


class QF_FB_BN_BAdj(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(QF_FB_BN_BAdj, self).__init__()
        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=False)
        self.momentum = momentum
        self.x_l_0_5 = Parameter(torch.zeros(num_features), requires_grad=False)
        self.x_g_0_5 = Parameter(torch.zeros(num_features), requires_grad=False)

    def forward(self, x, training=True):
        if not training:
            x_1 = self.x_l_0_5 * (self.x_running_rot * (1 - x) + x)
            x_1 += self.x_g_0_5 * (self.x_running_rot * x)
        else:
            x = x.transpose(0, 1)
            x_sum = x.sum(-1)
            x_mean = x_sum / x.shape[-1]
            self.x_l_0_5[:] = ((x_mean <= 0.5).float())
            self.x_g_0_5[:] = ((x_mean > 0.5).float())
            y = self.x_l_0_5 * ((0.5 - x_mean) / (1 - x_mean))
            y += self.x_g_0_5 * (0.5 / x_mean)

            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * y
            x = x.transpose(0, 1)
            x_1 = self.x_l_0_5 * (y * (1 - x) + x)
            x_1 += self.x_g_0_5 * (y * x)
        return x_1

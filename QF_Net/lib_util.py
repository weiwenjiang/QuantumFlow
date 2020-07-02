import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math
import shutil
import os
import sys

def modify_target_ori(target,interest_num):
    for j in range(len(target)):
        for idx in range(len(interest_num)):
            if target[j] == interest_num[idx]:
                target[j] = idx
                break
    new_target = torch.zeros(target.shape[0], len(interest_num))
    for i in range(target.shape[0]):
        one_shot = torch.zeros(len(interest_num))
        one_shot[target[i].item()] = 1
        new_target[i] = one_shot.clone()
    return target, new_target

def modify_target(target,interest_num):
    new_target = torch.zeros(target.shape[0], len(interest_num))
    for i in range(target.shape[0]):
        one_shot = torch.zeros(len(interest_num))
        one_shot[target[i].item()] = 1
        new_target[i] = one_shot.clone()
    return target, new_target

def select_num(dataset, interest_num):
    labels = dataset.targets  # get labels
    labels = labels.numpy()
    idx = {}
    for num in interest_num:
        idx[num] = np.where(labels == num)
    fin_idx = idx[interest_num[0]]
    for i in range(1, len(interest_num)):
        fin_idx = (np.concatenate((fin_idx[0], idx[interest_num[i]][0])),)
    fin_idx = fin_idx[0]
    dataset.targets = labels[fin_idx]
    dataset.data = dataset.data[fin_idx]
    dataset.targets, _ = modify_target_ori(dataset.targets, interest_num)
    return dataset

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.tar')
        shutil.copyfile(filename, bestname)


class BinarizeF(Function):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output
    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class ClipF(Function):
    @staticmethod
    def forward(ctx, input):
        output = input.clone().detach()
        output[input >= 1] = 1
        output[input <= 0] = 0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input >= 1] = 0
        grad_input[input <= 0] = 0
        return grad_input


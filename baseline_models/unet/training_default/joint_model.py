import torch
import torch.nn as nn

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
from layers import (
    Conv1d,
    GroupNorm,
    Linear,
    UNetBlock,
    UNetBlock_noatten,
    UNetBlock_atten,
    ScriptableAttentionOp,
)
from torch.nn.functional import silu
from typing import List


from climsim_utils.data_utils import *

from conflictfree.grad_operator import ConFIG_update


class JointModel(nn.Module):
    def __init__(self, deterministic_model, res_model):
        """
        deterministic_model, res_model: already-instantiated nn.Module objects
        """
        super().__init__()
        self.deterministic_model = deterministic_model
        self.res_model = res_model

    def forward(self, input, target):
        output = self.deterministic_model(input)
        
        residual = target - output
        residual = residual.to(output.device)
        
        #set the sigma based on parameters -- CHANGE THIS LATER
        P_mean = -1.2
        P_std = 1.2

        # Batch size
        batch_size = residual.shape[0]

        # Sample log-normal σ
        sigma = torch.exp(
            P_mean + P_std * torch.randn(batch_size, device=output.device)
        )
        
        predicted_residual = self.res_model(residual,sigma)

        return output, residual, predicted_residual

    def compute_loss(self, criterion, output, target, predicted_residual, residual):
        """
        Customize loss combination here.
        """
        
        deterministic_loss = criterion(output, target)
        res_loss = criterion(predicted_residual,residual)

        # Example weighted sum
        return deterministic_loss, res_loss

    def backward(self, deterministic_loss, res_loss, joint_optimizer):
        """
        Custom backward logic.
        """
        #gather all gradient parameters from both models
        params_a = [p for p in self.deterministic_model.parameters() if p.requires_grad]
        params_b = [p for p in self.res_model.parameters() if p.requires_grad]
        all_params = params_a + params_b

        #collect the gradients over both models according to the determinsitic loss
        grads_det = torch.autograd.grad(
            deterministic_loss, all_params, retain_graph=True, allow_unused=True
        )
        #flatten the gradients, setting any None gradients (like those in res model) to zero
        flat_grads_det = torch.cat([
            g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
            for g, p in zip(grads_det, all_params)
        ])
        
        #collect the gradients over both models according to the residual loss
        grads_res = torch.autograd.grad(
            res_loss, all_params, retain_graph=False, allow_unused=True
        )
        #flatten the gradients, setting any None gradients to zero
        flat_grads_res = torch.cat([
            g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
            for g, p in zip(grads_res, all_params)
        ])
        
        grads = [flat_grads_det, flat_grads_res]
        
        g_config=ConFIG_update(grads) # calculate the conflict-free direction
        joint_optimizer.zero_grad()
        data_utils.joint_apply_gradient_vector(self.deterministic_model, self.res_model,g_config) # set the conflict-free direction to the network

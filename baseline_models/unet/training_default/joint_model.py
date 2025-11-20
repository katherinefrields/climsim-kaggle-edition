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
    def __init__(self, model_a, model_b):
        """
        model_a, model_b: already-instantiated nn.Module objects
        """
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, input, target):
        """
        Customize however you want.
        Example: run both models on same input,
        or feed output from one into the other.
        """
        output = self.model_a(input)
        
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
        
        
        predicted_residual = self.model_b(residual,sigma)

        return output, residual, predicted_residual

    def compute_loss(self, criterion, output, target, predicted_residual, residual):
        """
        Customize loss combination here.
        """
        # Example losses
        deterministic_loss = criterion(output, target)
        
        res_loss = criterion(predicted_residual,residual)

        # Example weighted sum
        return deterministic_loss, res_loss

    def backward(self, deterministic_loss, res_loss, joint_optimizer):
        """
        Custom backward logic.
        """
        #joint_optimizer.zero_grad()
        #deterministic_loss.backward(retain_graph=True)
        
        params_a = [p for p in self.model_a.parameters() if p.requires_grad]
        params_b = [p for p in self.model_b.parameters() if p.requires_grad]
        all_params = params_a + params_b

        grads_det = torch.autograd.grad(
            deterministic_loss, all_params, retain_graph=True, allow_unused=True
        )
        flat_grads_det = torch.cat([
            g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
            for g, p in zip(grads_det, params_a)
        ])
        
        #the res gradients for deterministic grad will all be zero, since they don't affect the deterministic loss
        #deterministic_grad = data_utils.joint_get_gradient_vector(self.model_a, self.model_b, none_grad_mode="zero")
        
        #joint_optimizer.zero_grad()
        #res_loss.backward()
        grads_res = torch.autograd.grad(
            res_loss, all_params, retain_graph=False, allow_unused=True
        )
        flat_grads_res = torch.cat([
            g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
            for g, p in zip(grads_res, params_b)
        ])
        
        #res_grad = data_utils.joint_get_gradient_vector(self.model_a, self.model_b, none_grad_mode="zero")
        grads = [flat_grads_det, flat_grads_res]
        
        g_config=ConFIG_update(grads) # calculate the conflict-free direction
        joint_optimizer.zero_grad()
        data_utils.joint_apply_gradient_vector(self.model_a, self.model_b,g_config) # set the conflict-free direction to the network

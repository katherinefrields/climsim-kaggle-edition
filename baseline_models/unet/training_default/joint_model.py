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
        joint_optimizer.zero_grad()
        deterministic_loss.backward(retain_graph=True)
        
        #the res gradients for deterministic grad will all be zero, since they don't affect the deterministic loss
        deterministic_grad = data_utils.joint_get_gradient_vector(model, res_model, none_grad_mode="zero")
        
        joint_optimizer.zero_grad()
        res_loss.backward()
        res_grad = data_utils.joint_get_gradient_vector(model, res_model, none_grad_mode="zero")
        
        grads = [deterministic_grad, res_grad]
        

        g_config=ConFIG_update(grads) # calculate the conflict-free direction
        data_utils.joint_apply_gradient_vector(model, res_model,g_config) # set the conflict-free direction to the network

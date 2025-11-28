import torch
import torch.nn as nn

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus

from torch.nn.functional import silu
from typing import List


from climsim_utils.data_utils import *

from conflictfree.grad_operator import ConFIG_update


    
    
class JointModel(nn.Module):
    def __init__(self, deterministic_model, res_model, res_std_path, res_mean_path, preds_std_path, preds_mean_path):
        """
        deterministic_model, res_model: already-instantiated nn.Module objects
        """
        super().__init__()
        self.deterministic_model = deterministic_model
        self.res_model = res_model
        
        res_std = torch.load(res_std_path).to(deterministic_model.device)
        self.res_std = res_std.to(torch.float32)

        res_mean = torch.load(res_mean_path).to(deterministic_model.device)
        self.res_mean = res_mean.to(torch.float32)

        preds_std = torch.load(preds_std_path).to(deterministic_model.device)
        self.preds_std = preds_std.to(torch.float32)

        preds_mean = torch.load(preds_mean_path).to(deterministic_model.device)
        self.preds_mean = preds_mean.to(torch.float32)
        


    def forward(self, input, target):
        output = self.deterministic_model(input)
        
        residual = target - output
        residual = residual.to(output.device)
        
        #set the sigma based on parameters -- CHANGE THIS LATER
        
         #======Normalize input and condition data======
        normalized_residual = (residual - self.res_mean)/((self.res_std+ 1e-8) * 0.5)
        condition_input = (output - self.preds_mean)/((self.preds_std + 1e-8) * 0.5)
        
        ''' #Batch size
        P_mean = -1.2
        P_std = 1.2
        batch_size = residual.shape[0]

        # Sample log-normal σ
        sigma = torch.exp(
            P_mean + P_std * torch.randn(batch_size, device=output.device)
        )
        '''
       
        normalized_predicted_residual, weight = self.res_model(normalized_residual,self.res_std, self.res_mean, self.preds_std, self.preds_mean, condition = condition_input)
        predicted_residual = normalized_predicted_residual*((self.res_std+ 1e-8) * 0.5) + self.mean_data
        
        #predicted_residual is scaled back to original data space
        return output, residual, predicted_residual, normalized_residual, normalized_predicted_residual, weight

    def compute_loss(self, criterion, output, target, predicted_residual, residual, weight):
        """
        Customize loss combination here.
        """
        
        deterministic_loss = criterion(output, target)
        res_loss = (weight * ((predicted_residual - residual) ** 2)).mean()

        print(f'deterministic loss: {deterministic_loss.item()}, residual loss: {res_loss.item()}')
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

        if len(params_a)==0:
            grads_res = torch.autograd.grad(
                res_loss, all_params, retain_graph=False, allow_unused=True
            )
            #flatten the gradients, setting any None gradients to zero
            flat_grads_res = torch.cat([
                g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
                for g, p in zip(grads_res, all_params)
            ])
                
            grads = [flat_grads_res]
        else:
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

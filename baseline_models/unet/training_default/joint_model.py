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
    def __init__(self, deterministic_model, res_model, res_std_path, res_mean_path, 
                 preds_std_path, preds_mean_path,input_profile_num, 
                 input_scalar_num, vertical_level_num=60, img_resolution=64, sigma_data = .5):
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
        
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.vertical_level_num = vertical_level_num
        self.input_padding = (4,0)
        self.sigma_data = .5


    def forward(self, input, target):
        #output is shape (B, C*L)
        output = self.deterministic_model(input)
        #CHANGED THIS NAME TEMPORARILY!!!!!!
        
        residual = target - output
        residual = residual.to(output.device)
        
        #residual_a = target - output
        #residual = torch.zeros_like(residual_a)
        
        #======Normalize input and condition data======
        normalized_residual = ((residual - self.res_mean)/((self.res_std+ 1e-8)))*.5
        condition_input = ((output - self.preds_mean)/((self.preds_std + 1e-8)))*.5
    
        
        #=====Reshape Residaul=====
        #when you train your own model, have it 
        x_profile = normalized_residual[:,:self.input_profile_num*self.vertical_level_num]
        x_scalar = normalized_residual[:,self.input_profile_num*self.vertical_level_num:]
        
        # reshape x_profile to (batch, input_profile_num, levels)
        x_profile = x_profile.reshape(-1, self.input_profile_num, self.vertical_level_num)
        
        # broadcast x_scalar to (batch, input_scalar_num, levels)
        x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, self.vertical_level_num)
        
        #concatenate x_profile, x_scalar, x_loc to (batch, input_profile_num+input_scalar_num, levels)
        x = torch.cat((x_profile, x_scalar), dim=1)
        
        x = torch.nn.functional.pad(x, self.input_padding, "constant", 0.0)
        #x is (B, C, L)
        
        #=====Reshape Condition=====
        condition_profile = condition_input[:,:self.input_profile_num*self.vertical_level_num]
        condition_scalar = condition_input[:,self.input_profile_num*self.vertical_level_num:]
        
        # reshape x_profile to (batch, input_profile_num, levels)
        condition_profile = condition_profile.reshape(-1, self.input_profile_num, self.vertical_level_num)
        
        # broadcast x_scalar to (batch, input_scalar_num, levels)
        condition_scalar = condition_scalar.unsqueeze(2).expand(-1, -1, self.vertical_level_num)
        
        #concatenate x_profile, x_scalar, x_loc to (batch, input_profile_num+input_scalar_num, levels)
        condition_cat = torch.cat((condition_profile, condition_scalar), dim=1)
        condition_cat = torch.nn.functional.pad(condition_cat, self.input_padding, "constant", 0.0)
        #Condition is (B, C, L)
        
        
        ''' #Batch size
        P_mean = -1.2
        P_std = 1.2
        batch_size = residual.shape[0]

        # Sample log-normal σ
        sigma = torch.exp(
            P_mean + P_std * torch.randn(batch_size, device=output.device)
        )
        '''
       
        P_mean = -4.0
        P_std = 1.2
        batch_size = residual.shape[0]
        
        #trying this rand shape. it was different in the EDM Sampler -->
        #rnd_normal = torch.randn(x.shape, device=x.device)
        
        #apply the same noise to all features in the batch
        rnd_normal = torch.randn([batch_size,1,  1], device=residual.device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        
        #======Noises Residual======
        n = torch.randn_like(x) * sigma
        noised_residual = x + n
        #noised_residual = torch.likes(noised_residual)
        
        
        # weight per batch element according to the noise that was added to it
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #x is input noise image, D_x is predicted denoised image (B, C, L), y is predicted denoised image shape (B, C*L)
        D_x,  y = self.res_model(noised_residual,sigma, condition = condition_cat)
        
        #predicted_residual is scaled back to original data space
        return output, x,  D_x, y, weight

    def compute_loss(self, criterion, output, target, x, D_x, weight):
        """
        Customize loss combination here.
        """
        deterministic_loss = criterion(output, target)
        res_loss =  (weight*((x - D_x) ** 2)).mean() # calculate over C and L features
        #print(f'predicted value is {D_x}')
        #print(f'true value is {x}')
        #res_loss = (unweighted_res_loss * weight).mean()  # weighted residual loss
        #print(f'deterministic loss: {deterministic_loss.item()}, residual loss: {res_loss.item()}')
        # Example weighted sum
        return deterministic_loss, res_loss

    def backward(self, deterministic_loss, res_loss, joint_optimizer):
         # 1. Zero all grads
        joint_optimizer.zero_grad()

        # 2. Block all gradients from deterministic model
        deterministic_loss = deterministic_loss.detach()

        # 3. Backprop only through res_model
        res_loss.backward()

        # 4. Update only res_model parameters
        for p in self.deterministic_model.parameters():
            p.grad = None
        
        

            
        
        

    '''def backward(self, deterministic_loss, res_loss, joint_optimizer):
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
        data_utils.joint_apply_gradient_vector(self.deterministic_model, self.res_model,g_config) # set the conflict-free direction to the network'''
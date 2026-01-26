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
    def __init__(self, deterministic_model, res_model, res_std, res_mean, 
                 preds_std, preds_mean,
                 input_profile_num, 
                 input_scalar_num, 
                 target_profile_num, 
                 target_scalar_num, 
                 condition_channel_num,
                 vertical_level_num=60, 
                 img_resolution=64, sigma_data = .5, 
                 p_mean = -4.0, p_std=1.2):
        """
        deterministic_model, res_model: already-instantiated nn.Module objects
        """
        super().__init__()
        self.deterministic_model = deterministic_model
        self.res_model = res_model
        
        self.res_std = res_std
        self.res_mean = res_mean
        self.preds_std = preds_std
        self.preds_mean = preds_mean
        
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        
        self.condition_channel_num = condition_channel_num
        
        self.vertical_level_num = vertical_level_num
        self.input_padding = (4,0)
        self.sigma_data = .5
        
        self.p_mean = p_mean
        self.p_std = p_std

     #output is (B, C*L)
    #normalized true residual is (B, C*L)
    #normalized_predicted_residual is (B, C*L)
    #denormalized_residual (B, C, L)
    #denormalized_predicted_residual (B, C, L)
    #weight is the weight associated with the batch for the loss function
    def forward(self, input, target):
        #output is shape (B, C*L)

        # (B, C*L) --> (B, C, L)
        input = self.reshape_input(input)
        target = self.reshape_target(target)
        
        #=====Calculate Residual=====
        #output shape is (B, C, L), scalar values are all expanded mean value across levels
        output = self.deterministic_model(input)
        
        residual = target - output
        residual = residual.to(output.device)
        
        print(f'location 1 residual requires grad = {residual.requires_grad}')
        #residual = torch.zeros_like(residual)
        
        #residual = self.reverse_reshape_target(residual)
        #residual = self.reshape_target(residual)
        
        #======Normalize input and condition data======
        #normalized_residual = residual
        normalized_residual = ((residual - self.res_mean)/((self.res_std+ 1e-8)))*.5
        condition_output = ((output - self.preds_mean)/((self.preds_std + 1e-8)))*.5
    
        
        #normalized_residual = self.reverse_reshape_target(normalized_residual)
        #normalized_residual = self.reshape_target(normalized_residual)
        
        '''
        #Batch size
        P_mean = -1.2
        P_std = 1.2
        batch_size = residual.shape[0]

        # Sample log-normal σ
        sigma = torch.exp(
            P_mean + P_std * torch.randn(batch_size, device=output.device)
        )
        '''
       
        batch_size = residual.shape[0]
        
        #trying this rand shape. it was different in the EDM Sampler -->
        #rnd_normal = torch.randn(x.shape, device=x.device)
        
        #apply the same noise to all features in the batch
        rnd_normal = torch.randn([batch_size,1,  1], device=residual.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        
        #======Noises Residual======
        n = torch.randn_like(normalized_residual) * sigma
        noised_residual = normalized_residual + n
        
        print(f'location 2 noised_residual requires grad = {noised_residual.requires_grad}')
        
        
        # weight per batch element according to the noise that was added to it
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        if weight.flatten().mean() > 100:
            print (f'weight is exploding')
        #this was used for classifier free guidance
        # ---- INSERT CONDITIONING DROPOUT HERE ----
        #drop_mask = (torch.rand(batch_size, 1, 1, device=condition_cat.device) < 0.1)
        #condition_cat = condition_cat * (~drop_mask)  # OR (1 - drop_mask.float())

    
        #shape (B,C,L)
        normalized_predicted_residual = self.res_model(noised_residual,sigma, condition = condition_output)
        
        print(f'location 3 normalized_predicted_residual requires grad = {normalized_predicted_residual.requires_grad}')
        #print(f'self.res_std mean is {self.res_std.flatten().mean()}')
        #=====Reshape Predicted residual=====
        #reshape true residual and predicted residual back to (B, C*L)
        #denormalized_residual = normalized_residual
        #denormalized_predicted_residual = normalized_predicted_residual
        
        denormalized_residual = normalized_residual / .5 * (self.res_std+ 1e-8) + self.res_mean
        denormalized_predicted_residual = normalized_predicted_residual / .5 * (self.res_std+ 1e-8) + self.res_mean
        
        #(B,C,L) --> (B, C*L) 
        denormalized_residual = self.reverse_reshape_target(denormalized_residual)
        denormalized_predicted_residual = self.reverse_reshape_target(denormalized_predicted_residual)
        
        print(f'location 3 denormalized_predicted_residual requires grad = {denormalized_predicted_residual.requires_grad}')
        print(f'location 4 denormalized_residual requires grad = {denormalized_residual.requires_grad}')
        
        
        #(B,C,L) --> (B, C*L) 
        normalized_residual = self.reverse_reshape_target(normalized_residual)
        normalized_predicted_residual = self.reverse_reshape_target(normalized_predicted_residual)
    
        #normalized_residual = self.reverse_reshape_target(normalized_residual)
        #normalized_predicted_residual = self.reverse_reshape_target(normalized_predicted_residual)
        output = self.reverse_reshape_target(output)
        target = self.reverse_reshape_target(target)
        #output is denormalized
        
        if normalized_predicted_residual.flatten().mean() > 100:
            print (f'normalized_predicted_residual is exploding')
            
        if normalized_residual.flatten().mean() > 100:
            print (f'normalized_residual is exploding')
            
        if denormalized_predicted_residual.flatten().mean() > 100:
            print (f'denormalized_predicted_residual is exploding')
            
        if denormalized_residual.flatten().mean() > 100:
            print (f'denormalized_residual is exploding')
            
        return output, target, denormalized_predicted_residual, denormalized_residual, normalized_predicted_residual, normalized_residual, weight

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
         
        for p in self.res_model.parameters():
            print(f'p_grad is {p.grad}')
            
        joint_optimizer.zero_grad()

        # 2. Block all gradients from deterministic model
        deterministic_loss = deterministic_loss.detach()
        
        # 4. Update only res_model parameters
        for p in self.deterministic_model.parameters():
            p.requires_grad = False
            p.grad = None

        # 3. Backprop only through res_model
        res_loss.backward()
            
        
    
    #reshapes target from (B,C*L ) to (B, C, L)
    def reshape_target(self, target):
        #=====Reshape Target Condition=====
        #FINISH RESHAPING TARGET
        target_profile = target[:,:self.target_profile_num*self.vertical_level_num]
        target_scalar = target[:,self.target_profile_num*self.vertical_level_num:]
        
        # reshape x_profile to (batch, input_profile_num, levels)
        target_profile = target_profile.reshape(-1, self.target_profile_num, self.vertical_level_num)
        
        # broadcast x_scalar to (batch, target_scalar_num, levels)
        target_scalar = target_scalar.unsqueeze(2).expand(-1, -1, self.vertical_level_num)
        
        #concatenate x_profile, x_scalar, x_loc to (batch, input_profile_num+target_scalar_num, levels)
        target = torch.cat((target_profile, target_scalar), dim=1)
        
        target = torch.nn.functional.pad(target, self.input_padding, "constant", 0.0)
        
        return target
    
    #reshapes input from (B,C*L ) to (B, C, L)
    def reshape_input(self, input):
        #=====Reshape Input=====
        # split x into x_profile and x_scalar
        x_profile = input[:,:self.input_profile_num*self.vertical_level_num]
        x_scalar = input[:,self.input_profile_num*self.vertical_level_num:]

        # reshape x_profile to (batch, input_profile_num, levels)
        x_profile = x_profile.reshape(-1, self.input_profile_num, self.vertical_level_num)
        # broadcast x_scalar to (batch, input_scalar_num, levels)
        x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, self.vertical_level_num)

        #concatenate x_profile, x_scalar, x_loc to (batch, input_profile_num+input_scalar_num, levels)
        x = torch.cat((x_profile, x_scalar), dim=1)
        
        # pads the beginning of levels so that levels = seq_resolution (which by default is 64)
        input = torch.nn.functional.pad(x, self.input_padding, "constant", 0.0)
        
        return input
    
    #reshapes target from (B,C,L) to (B, C*L)
    def reverse_reshape_target(self, target):
        #=====Reshape Target Condition=====
        y_profile = target[:,:self.target_profile_num,self.input_padding[0]:]
        y_scalar = target[:,self.target_profile_num:,self.input_padding[0]:]
        
        #print(f'y_profile shape is {y_profile.shape}')
        #print(f'y_scalar shape is {y_scalar.shape}')

        y_scalar = y_scalar.mean(dim=2)
        y_profile = y_profile.reshape(-1, self.target_profile_num*self.vertical_level_num)
        #print(f'before concat y_profile shape is {y_profile.shape} and y_scalar shape is {y_scalar.shape}')
        y = torch.cat((y_profile, y_scalar), dim=1)
        return y
    
    
   
        

            
        
        

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
import torch
import torch.nn as nn

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
import math

from torch.nn.functional import silu
from typing import List
from torch.distributions.studentT import StudentT

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

from climsim_utils.data_utils import *

from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector


@dataclass
class JointModelMetaData(modulus.ModelMetaData):
    """JointModel meta data"""

    name: str = "JointModel"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False
    
class JointModel(modulus.Module):
    def __init__(self, deterministic_model, res_model, res_std, res_mean, 
                 preds_std, preds_mean,
                 input_profile_num, 
                 input_scalar_num, 
                 target_profile_num, 
                 target_scalar_num, 
                 condition_channel_num,
                 condition_type = 'input_output',
                 condtition_location = 'embedding',
                 vertical_level_num=60,
                 img_resolution=64, sigma_data = .5,
                 p_mean = -4.0, p_std=1.2, nu = 3, t_sampling = False,
                 amp_mode = False):
        """
        deterministic_model, res_model: already-instantiated nn.Module objects
        """
        super().__init__(meta=JointModelMetaData)
        self.deterministic_model = deterministic_model
        self.res_model = res_model
        #self.res_std = res_std
        #self.res_mean = res_mean
        #self.preds_std = preds_std
        #self.preds_mean = preds_mean
        
        self.register_buffer("res_std", torch.as_tensor(res_std, dtype=torch.float32))
        self.register_buffer("res_mean", torch.as_tensor(res_mean, dtype=torch.float32))
        self.register_buffer("preds_std", torch.as_tensor(preds_std, dtype=torch.float32))
        self.register_buffer("preds_mean", torch.as_tensor(preds_mean, dtype=torch.float32))
        
        
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        
        self.condition_channel_num = condition_channel_num
        self.condition_type = condition_type
        self.condition_location = condtition_location
        
        self.vertical_level_num = vertical_level_num
        self.input_padding = (4,0)
        self.sigma_data = sigma_data
        
        self.p_mean = p_mean
        self.p_std = p_std
        self.nu = nu
        
        self.t_sampling = t_sampling

        #loops through all modules and their layers and sets the amp mode to true if the layer has an amp mode attribute. this is necessary for the layers in the res_model to be in amp mode, which is important for memory efficiency and speed.
        if amp_mode:
            for m in self.modules():
                if hasattr(m, 'amp_mode'):
                    m.amp_mode = True
        
        '''self.res_affine = nn.Sequential(
            nn.LayerNorm([self.target_profile_num + self.target_scalar_num, 64], elementwise_affine = False)
            #nn.Linear(64,64)
        )'''

        '''self.cond_affine = nn.Sequential(
            nn.LayerNorm([self.target_profile_num + self.target_scalar_num, 64],  elementwise_affine = False)
            #nn.Linear(64,64) 
            )'''



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
        output, latent_output = self.deterministic_model(input)
        
        residual = target - output
        residual = residual.to(output.device)
        
        
        #======Normalize input and condition data======
        #normalized_residual = residual
        safe_std = torch.clamp(self.res_std, min=1e-2)
        normalized_residual = ((residual)/((safe_std+ 1e-8)))*.5
        condition_output = ((output - self.preds_mean)/((self.preds_std + 1e-8)))*.5
        
        latent_condition = torch.cat((input, condition_output), dim=1)
        condition_data = latent_condition  # default; overwritten below as needed
        if self.condition_location == 'front':
            if self.condition_type == 'input_output':
                condition_data = latent_condition
        elif self.condition_location == 'embedding':
            if self.condition_type == 'input_output':
                latent_condition = torch.cat((input, condition_output), dim=1)
            else:
                latent_condition = latent_output
            condition_data = latent_condition.reshape(latent_condition.shape[0], -1)
        elif self.condition_location == 'middle' or self.condition_location == 'cross':
            if self.condition_type == 'input_output':
                latent_condition = torch.cat((input, condition_output), dim=1)
            condition_data = latent_condition
            #print(latent_condition.shape)
        
        B, C, L = normalized_residual.shape
        
        #use a seperate sigma for each batch element, but the same sigma across all features in the batch element
        rnd_normal = torch.randn([B, 1, 1], device=residual.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()   # no scaling applied
        
        
        if self.t_sampling == False:
            n = torch.randn_like(normalized_residual) * sigma
            noised_residual = normalized_residual + n
        else:
            # Gaussian base noise
            z = torch.randn((B, C, L), device=residual.device)
            nu = torch.tensor([self.nu]).to(residual.device)
            # One kappa per sample — Chi2(df=nu) = sum of nu squared standard normals
            
            #old, removed by Claude
            #kappa = torch.distributions.Chi2(df=nu).sample((B,)).to(residual.device)

            nu_int = int(self.nu)
            kappa = (torch.randn(B, nu_int, device=residual.device) ** 2).sum(dim=1)
            kappa = (kappa / nu).view(B, 1, 1)

            # Student‑t noise
            t_noise = z / torch.sqrt(kappa)

            # Normalize to unit variance (REQUIRED for EDM)
            #t_noise = t_noise * math.sqrt((self.nu - 2) / self.nu)

            # Apply sigma
            sigma = sigma.view(B, 1, 1)
            n = t_noise * sigma
            print(f'n shape is {n.shape}')
            print(f'normalized_residual shape is {normalized_residual.shape}')
            noised_residual = normalized_residual + n
            sigma = sigma*torch.sqrt(nu/(nu-2))
            
        # weight per batch element according to the noise that was added to it
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        '''if weight.flatten().mean() > 100:
            print (f'weight is exploding: {weight.flatten().mean()}')'''
        #this was used for classifier free guidance
        # ---- INSERT CONDITIONING DROPOUT HERE ----
        #drop_mask = (torch.rand(batch_size, 1, 1, device=condition_cat.device) < 0.1)
        #condition_cat = condition_cat * (~drop_mask)  # OR (1 - drop_mask.float())

    
        #shape (B,C,L)
        normalized_predicted_residual = self.res_model(noised_residual,sigma, condition = condition_data)
        
        
        denormalized_residual = normalized_residual / .5 * (safe_std + 1e-8)
        denormalized_predicted_residual = normalized_predicted_residual / .5 * (safe_std + 1e-8)
        
        output = self.reverse_reshape_target(output)
        target = self.reverse_reshape_target(target)
        #output is denormalized
        
        return output, target, denormalized_predicted_residual, denormalized_residual, normalized_predicted_residual, normalized_residual, weight


    @torch.jit.export
    def inference(self, input: torch.Tensor) -> torch.Tensor:
        #output is shape (B, C*L)
        # (B, C*L) --> (B, C, L)
        input = self.reshape_input(input)

        #=====Calculate Residual=====
        #output shape is (B, C, L), scalar values are all expanded mean value across levels
        output, latent_output = self.deterministic_model(input)

        #======Normalize input and condition data======
        safe_std = torch.clamp(self.res_std, min=1e-2)
        condition_output = ((output - self.preds_mean)/((self.preds_std + 1e-8)))*.5

        latent_condition = torch.cat((input, condition_output), dim=1)
        condition_data = latent_condition  # default; overwritten below as needed
        if self.condition_location == 'front':
            if self.condition_type == 'input_output':
                condition_data = latent_condition
        elif self.condition_location == 'embedding':
            if self.condition_type == 'input_output':
                latent_condition = torch.cat((input, condition_output), dim=1)
            else:
                latent_condition = latent_output
            condition_data = latent_condition.reshape(latent_condition.shape[0], -1)
        elif self.condition_location == 'middle' or self.condition_location == 'cross':
            if self.condition_type == 'input_output':
                latent_condition = torch.cat((input, condition_output), dim=1)
            condition_data = latent_condition
            #print(latent_condition.shape)

        latents = torch.randn_like(output)
        res = self.res_model.edm_sampler(latents, condition_input = condition_data, sigma_min=0.1, sigma_max=45.0, rho = 7.0, num_steps = 18) #maybe have to remove the last 4 meaningless levels??
            
        denormalized_predicted_residual = (res/.5)*((safe_std+ 1e-8))
        #(B,C,L) --> (B, C*L)
        reshaped_res = self.reverse_reshape_target(denormalized_predicted_residual)
        #(B,C,L) --> (B, C*L)
        output = self.reverse_reshape_target(output)
        joint_pred = output + reshaped_res
        
        #condition_data = torch.cat((latent_condition, input), dim=1)
        #normalized_residual = self.reverse_reshape_target(normalized_residual)
        #normalized_residual = self.reshape_target(normalized_residual)
        '''
        B,C,L = residual.shape[0],residual.shape[1],residual.shape[2]
        
        rnd_normal = torch.randn([B,1,1], device=residual.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()   # no scaling applied
        t_distribution = StudentT(df=self.nu, loc=0, scale=sigma)
        n = t_distribution.sample(sample_shape = torch.Size([B,C,L])).squeeze(-1)
        '''
        
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
        '''
            batch_size = residual.shape[0]
            
            #trying this rand shape. it was different in the EDM Sampler -->
            #rnd_normal = torch.randn(x.shape, device=x.device)
            nu = self.nu

            # apply the same noise level σ to all features in the batch
            rnd_normal = torch.randn([batch_size, 1, 1], device=residual.device)
            sigma = (rnd_normal * self.p_std + self.p_mean).exp()   # no scaling applied

            # --- Student‑t noise (Pandey et al. 2024) ---
            # Gaussian base noise
            z = torch.randn_like(normalized_residual)

            # One kappa per sample (correct multivariate Student‑t)
            B = normalized_residual.shape[0]
            kappa = torch.distributions.Chi2(df=nu).sample((B,)).to(normalized_residual.device)
            kappa = (kappa / nu).view(B, 1, 1)   # broadcast to (B, C, L)

            # Student‑t noise
            t_noise = z / torch.sqrt(kappa)
            
            

            # Apply σ
            n = t_noise * sigma
            # --------------------------------------------

            noised_residual = normalized_residual + n'''
        
        return joint_pred
    
    @torch.no_grad()
    def sample_ensemble(self, input: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """
        Draw num_samples from the diffusion model and return ensemble predictions.

        Returns (B, num_samples, C*L) — ready for EnsembleCRPSLoss.
        """
        input = self.reshape_input(input)
        output, latent_output = self.deterministic_model(input)

        safe_std = torch.clamp(self.res_std, min=1e-2)
        condition_output = ((output - self.preds_mean) / (self.preds_std + 1e-8)) * 0.5

        if self.condition_location == 'embedding':
            if self.condition_type == 'input_output':
                latent_condition = torch.cat((input, condition_output), dim=1)
            else:
                latent_condition = latent_output
            condition_data = latent_condition.reshape(latent_condition.shape[0], -1)
        elif self.condition_location == 'front' or self.condition_location == 'middle' or self.condition_location == 'cross':
            latent_condition = torch.cat((input, condition_output), dim=1)
            condition_data = latent_condition
        else:
            condition_data = torch.cat((input, condition_output), dim=1)

        samples = []
        for _ in range(num_samples):
            latents = torch.randn_like(output)
            res = self.res_model.edm_sampler(latents, condition_input=condition_data,
                                             sigma_min=0.1, sigma_max=45.0, rho=7.0, num_steps=18)
            denorm_res = (res / 0.5) * (safe_std + 1e-8)
            pred = self.reverse_reshape_target(output) + self.reverse_reshape_target(denorm_res)
            samples.append(pred)

        return torch.stack(samples, dim=1)  # (B, num_samples, C*L)

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

    def backward(self,res_loss, joint_optimizer):
         # 1. Zero all grads
         
        #for p in self.res_model.parameters():
        #    print(f'p_grad is {p.grad}')
            
        joint_optimizer.zero_grad(set_to_none=True)

        # 2. Block all gradients from deterministic model
        #deterministic_loss = deterministic_loss.detach()
        
        # 4. Update only res_model parameters

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
    
    
   
        

            
        
    def compute_joint_loss(self, criterion, output, target, x, D_x, weight):
        """
        Customize loss combination here.
        """
        
        predicted_residual = self.reverse_reshape_target(D_x)
        combined_output = output + predicted_residual
        deterministic_loss = criterion(combined_output, target)
        
        res_loss =  (weight*((x - D_x) ** 2)).mean() # calculate over C and L features
        #print(f'predicted value is {D_x}')
        #print(f'true value is {x}')
        #res_loss = (unweighted_res_loss * weight).mean()  # weighted residual loss
        #print(f'deterministic loss: {deterministic_loss.item()}, residual loss: {res_loss.item()}')
        # Example weighted sum
        return deterministic_loss, res_loss
    

    def joint_backward(self, deterministic_loss, res_loss, joint_optimizer):
        """
        Custom backward logic.
        """
        '''
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
            
            collect the gradients over both models according to the residual loss
            grads_res = torch.autograd.grad(
                res_loss, all_params, retain_graph=False, allow_unused=True
            )
            #flatten the gradients, setting any None gradients to zero
            flat_grads_res = torch.cat([
                g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
                for g, p in zip(grads_res, all_params)
            grads = [flat_grads_det, flat_grads_res]
        
            ])'''
        grads = []
        
        joint_optimizer.zero_grad()
        deterministic_loss.backward(retain_graph=True)
        grads.append(get_gradient_vector(self, none_grad_mode = 'zero'))
        
        joint_optimizer.zero_grad()
        res_loss.backward(retain_graph=True)
        grads.append(get_gradient_vector(self, none_grad_mode = 'zero'))
            
            
        g_config=ConFIG_update(grads) # calculate the conflict-free direction
        #joint_optimizer.zero_grad()
        apply_gradient_vector(self, g_config)
        #data_utils.joint_apply_gradient_vector(self.deterministic_model, self.res_model,g_config) # set the conflict-free direction to the network
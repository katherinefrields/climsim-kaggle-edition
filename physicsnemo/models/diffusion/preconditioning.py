#SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preconditioning schemes used in the paper"Elucidating the Design Space of
Diffusion-Based Generative Models".
"""

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, List, Literal, Tuple, Union

import numpy as np
import torch

from physicsnemo.models.diffusion.utils import _wrapped_property
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

network_module = importlib.import_module("physicsnemo.models.diffusion")


@dataclass
class EDMPrecondMetaData(ModelMetaData):
    """EDMPrecond meta data"""

    name: str = "EDMPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
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


class EDMPrecond(Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM)

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels (for both input and output). If your model
        requires a different number of input or output chanels,
        override this by passing either of the optional
        img_in_channels or img_out_channels args
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.0.
    sigma_max : float
        Maximum supported noise level, by default inf.
    sigma_data : float
        Expected standard deviation of the training data, by default 0.5.
    model_type :str
        Class name of the underlying model, by default "DhariwalUNet".
    img_in_channels: int
        Optional setting for when number of input channels =/= number of output
        channels. If set, will override img_channels for the input
        This is useful in the case of additional (conditional) channels
    img_out_channels: int
        Optional setting for when number of input channels =/= number of output
        channels. If set, will override img_channels for the output
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        input_profile_num: int = 9, # number of input profile variables
        input_scalar_num: int = 17, # number of input scalar variables
        vertical_level_num:int = 60,
        label_dim=0,
        use_fp16=False,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="DhariwalUNet",
        img_in_channels=None,
        img_out_channels=None,
        **model_kwargs,
    ):
        super().__init__(meta=EDMPrecondMetaData)
        self.img_resolution = img_resolution
        if img_in_channels is not None:
            img_in_channels = img_in_channels
        else:
            img_in_channels = img_channels
        if img_out_channels is not None:
            img_out_channels = img_out_channels
        else:
            img_out_channels = img_channels

        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.input_profile_num = input_profile_num # number of input profile variables
        self.input_scalar_num = input_scalar_num # number of input scalar variables
        self.vertical_level_num = vertical_level_num

        self.input_padding = (img_resolution - vertical_level_num,0)
        
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(
        self,
        x,
        sigma,
        condition=None,
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        #=====Cast to floats=====
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        
        #=====Reshape Input=====
        #levels are without padding
        #currently x(batch, target_profile_num*levels+target_scalar_num)
        
        x_profile = x[:,:self.input_profile_num*self.vertical_level_num]
        x_scalar = x[:,self.input_profile_num*self.vertical_level_num:]
        
        # reshape x_profile to (batch, input_profile_num, levels)
        x_profile = x_profile.reshape(-1, self.input_profile_num, self.vertical_level_num)
        
        # broadcast x_scalar to (batch, input_scalar_num, levels)
        x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, self.vertical_level_num)
        
        #concatenate x_profile, x_scalar, x_loc to (batch, input_profile_num+input_scalar_num, levels)
        x = torch.cat((x_profile, x_scalar), dim=1)
        
        x = torch.nn.functional.pad(x, self.input_padding, "constant", 0.0)
        
        #=====Reshape Condition=====
        #levels are without padding
        #currently x(batch, target_profile_num*levels+target_scalar_num)
        if condition != None:
            condition_profile = condition[:,:self.input_profile_num*self.vertical_level_num]
            condition_scalar = condition[:,self.input_profile_num*self.vertical_level_num:]
            
            # reshape x_profile to (batch, input_profile_num, levels)
            condition_profile = condition_profile.reshape(-1, self.input_profile_num, self.vertical_level_num)
            
            # broadcast x_scalar to (batch, input_scalar_num, levels)
            condition_scalar = condition_scalar.unsqueeze(2).expand(-1, -1, self.vertical_level_num)
            
            #concatenate x_profile, x_scalar, x_loc to (batch, input_profile_num+input_scalar_num, levels)
            condition_cat = torch.cat((condition_profile, condition_scalar), dim=1)
            
            condition_cat = torch.nn.functional.pad(condition_cat, self.input_padding, "constant", 0.0)
            
                
        #=====Class Conditioning=====
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        
        #=====Float use=====
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )
        
        #=====Compute Scaling Coefficients=====
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        #=====Model Conditioning=====
        arg = c_in * x

        if condition is not None:
            arg = torch.cat([arg, condition_cat], dim=1)

        #=====Predict Noise=====
        F_x = self.model(
            arg.to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        
        print(f'D_x shape is {D_x.shape}')

        y_profile = D_x[:,:self.input_profile_num,self.input_padding[0]:]
        y_scalar = D_x[:,self.input_profile_num:,self.input_padding[0]:]
        
        print(f'y_profile shape is {y_profile.shape}')
        print(f'y_scalar shape is {y_scalar.shape}')

        y_scalar = y_scalar.mean(dim=2)
        y_profile = y_profile.reshape(-1, self.input_profile_num*self.vertical_level_num)
        print(f'before concat y_profile shape is {y_profile.shape} and y_scalar shape is {y_scalar.shape}')
        y = torch.cat((y_profile, y_scalar), dim=1)
        
        return y

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)



'''
@dataclass
class VPPrecondMetaData(ModelMetaData):
    """VPPrecond meta data"""

    name: str = "VPPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
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


class VPPrecond(Module):
    """
    Preconditioning corresponding to the variance preserving (VP) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    beta_d : float
        Extent of the noise level schedule, by default 19.9.
    beta_min : float
        Initial slope of the noise level schedule, by default 0.1.
    M : int
        Original number of timesteps in the DDPM formulation, by default 1000.
    epsilon_t : float
        Minimum t-value used during training, by default 1e-5.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        M: int = 1000,
        epsilon_t: float = 1e-5,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__(meta=VPPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t: Union[float, torch.Tensor]):
        """
        Compute the sigma(t) value for a given t based on the VP formulation.

        The function calculates the noise level schedule for the diffusion process based
        on the given parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        t : Union[float, torch.Tensor]
            The timestep or set of timesteps for which to compute sigma(t).

        Returns
        -------
        torch.Tensor
            The computed sigma(t) value(s).
        """
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma: Union[float, torch.Tensor]):
        """
        Compute the inverse of the sigma function for a given sigma.

        This function effectively calculates t from a given sigma(t) based on the
        parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        sigma : Union[float, torch.Tensor]
            The sigma(t) value or set of sigma(t) values for which to compute the
            inverse.

        Returns
        -------
        torch.Tensor
            The computed t value(s) corresponding to the provided sigma(t).
        """
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def round_sigma(self, sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


@dataclass
class VEPrecondMetaData(ModelMetaData):
    """VEPrecond meta data"""

    name: str = "VEPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
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


class VEPrecond(Module):
    """
    Preconditioning corresponding to the variance exploding (VE) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__(meta=VEPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)
'''

'''
@dataclass
class EDMPrecondSuperResolutionMetaData(ModelMetaData):
    """EDMPrecondSuperResolution meta data"""

    name: str = "EDMPrecondSuperResolution"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
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


class EDMPrecondSuperResolution(Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM).

    This is a variant of `EDMPrecond` that is specifically designed for super-resolution
    tasks. It wraps a neural network that predicts the denoised high-resolution image
    given a noisy high-resolution image, and additional conditioning that includes a
    low-resolution image, and a noise level.

    Parameters
    ----------
    img_resolution : Union[int, Tuple[int, int]]
        Spatial resolution :math:`(H, W)` of the image. If a single int is provided,
        the image is assumed to be square.
    img_in_channels : int
        Number of input channels in the low-resolution input image.
    img_out_channels : int
        Number of output channels in the high-resolution output image.
    use_fp16 : bool, optional
        Whether to use half-precision floating point (FP16) for model execution,
        by default False.
    model_type : str, optional
        Class name of the underlying model. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
        Defaults to 'SongUNetPosEmbd'.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
    sigma_min : float, optional
        Minimum supported noise level, by default 0.0.
    sigma_max : float, optional
        Maximum supported noise level, by default inf.
    **model_kwargs : dict
        Keyword arguments passed to the underlying model `__init__` method.

    See Also
    --------
    For information on model types and their usage:
    :class:`~physicsnemo.models.diffusion.SongUNet`: Basic U-Net for diffusion models
    :class:`~physicsnemo.models.diffusion.SongUNetPosEmbd`: U-Net with positional embeddings
    :class:`~physicsnemo.models.diffusion.SongUNetPosLtEmbd`: U-Net with positional and lead-time embeddings

    Please refer to the documentation of these classes for details on how to call
    and use these models directly.

    Note
    ----
    References:
    - Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    - Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    # Classes that can be wrapped by this UNet class.
    _wrapped_classes = {
        "SongUNetPosEmbd",
        "SongUNetPosLtEmbd",
        "SongUNet",
        "DhariwalUNet",
    }

    # Arguments of the __init__ method that can be overridden with the
    # ``Module.from_checkpoint`` method. Here, since we use splatted arguments
    # for the wrapped model instance, we allow overriding of any overridable
    # argument of the wrapped classes.
    _overridable_args = set.union(
        *(
            getattr(getattr(network_module, cls_name), "_overridable_args", set())
            for cls_name in _wrapped_classes
        )
    )

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        sigma_data: float = 0.5,
        sigma_min=0.0,
        sigma_max=float("inf"),
        **model_kwargs: Any,
    ):
        super().__init__(meta=EDMPrecondSuperResolutionMetaData)

        # Validation
        if model_type not in self._wrapped_classes:
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                f"Must be one of: {', '.join(self._wrapped_classes)}"
            )

        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )  # TODO needs better handling
        self.scaling_fn = self._scaling_fn
        self.use_fp16 = use_fp16

    @property
    def use_fp16(self):
        """
        bool: Whether the model uses float16 precision.

        Returns
        -------
        bool
            True if the model is in float16 mode, False otherwise.
        """
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        """
        Set whether the model should use float16 precision.

        Parameters
        ----------
        value : bool
            If True, moves the model to torch.float16. If False, moves to torch.float32.

        Raises
        ------
        ValueError
            If `value` is not a boolean.
        """
        # NOTE: allow 0/1 values for older checkpoints
        if not (isinstance(value, bool) or value in [0, 1]):
            raise ValueError(
                f"`use_fp16` must be a boolean, but got {type(value).__name__}."
            )
        self._use_fp16 = value
        if value:
            self.to(torch.float16)
        else:
            self.to(torch.float32)

    @staticmethod
    def _scaling_fn(
        x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale input tensors by first scaling the high-resolution tensor and then
        concatenating with the low-resolution tensor.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution image of shape (B, C_lr, H, W).
        c_in : torch.Tensor
            Scaling factor of shape (B, 1, 1, 1).

        Returns
        -------
        torch.Tensor
            Scaled and concatenated tensor of shape (B, C_in+C_out, H, W).
        """
        return torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1)

    # Properties delegated to the wrapped model
    amp_mode = _wrapped_property(
        "amp_mode",
        "model",
        "Set to ``True`` when using automatic mixed precision.",
    )
    profile_mode = _wrapped_property(
        "profile_mode",
        "model",
        "Set to ``True`` to enable profiling of the wrapped model.",
    )

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass of the EDMPrecondSuperResolution model wrapper.

        This method applies the EDM preconditioning to compute the denoised image
        from a noisy high-resolution image and low-resolution conditioning image.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W). The number of
            channels `C_hr` should be equal to `img_out_channels`.
        img_lr : torch.Tensor
            Low-resolution conditioning image of shape (B, C_lr, H, W). The number
            of channels `C_lr` should be equal to `img_in_channels`.
        sigma : torch.Tensor
            Noise level of shape (B) or (B, 1) or (B, 1, 1, 1).
        force_fp32 : bool, optional
            Whether to force FP32 precision regardless of the `use_fp16` attribute,
            by default False.
        **model_kwargs : dict
            Additional keyword arguments to pass to the underlying model
            `self.model` forward method.

        Returns
        -------
        torch.Tensor
            Denoised high-resolution image of shape (B, C_hr, H, W).

        Raises
        ------
        ValueError
            If the model output dtype doesn't match the expected dtype.
        """
        # Concatenate input channels
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        if img_lr is None:
            arg = c_in * x
        else:
            arg = self.scaling_fn(x, img_lr, c_in)
        arg = arg.to(dtype)

        F_x = self.model(
            arg,
            c_noise.flatten(),
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float, List, torch.Tensor]
            Sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            Tensor representation of sigma values.

        See Also
        --------
        EDMPrecond.round_sigma
        """
        return EDMPrecond.round_sigma(sigma)

'''

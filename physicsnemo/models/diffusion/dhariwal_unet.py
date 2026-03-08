from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.nn.functional import silu

from physicsnemo.models.diffusion import (
    DiffConv1d,
    DiffLinear,
    DiffPositionalEmbedding,
    DiffUNetBlock,
    get_group_norm,
)
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module
import modulus


@dataclass
class MetaData(modulus.ModelMetaData):
    name: str = "DhariwalUNet"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False
    amp_gpu: bool = False
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class WrappedLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, is_block: bool):
        super().__init__()
        self.layer = layer
        self.is_block = is_block

    def forward(self, x, emb):
        if self.is_block:
            return self.layer(x, emb)
        else:
            return self.layer(x)


class DhariwalUNet(modulus.Module):
    def __init__(
        self,
        *,
        img_resolution: int,
        in_channels: int,
        out_channels: int,
        condition_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 192,
        channel_mult: List[int] = [1, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 3,
        attn_resolutions: List[int] = [32, 16, 8],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
        condition_location: str = "cross",
    ):
        super().__init__(meta=MetaData())
        self.label_dropout = label_dropout
        self.attn_resolutions = attn_resolutions
        self.channel_mult = channel_mult
        self.model_channels = model_channels
        emb_channels = model_channels * channel_mult_emb

        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
            resample_proj=True,
        )

        # ---------------- Mapping ----------------
        self.map_noise = DiffPositionalEmbedding(num_channels=model_channels)
        self.map_augment = (
            DiffLinear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
                **init_zero,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = DiffLinear(
            in_features=model_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = DiffLinear(
            in_features=emb_channels, out_features=emb_channels, **init
        )
        self.map_label = (
            DiffLinear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        self.condition_location = condition_location
        self.map_cond = torch.nn.Identity()
        self.cond_proj = torch.nn.Identity()

        self.in_channels = in_channels
        if self.condition_location == "embedding":
            self.map_cond = DiffLinear(
                in_features=condition_channels,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
            )
        elif condition_location == "middle":
            self.cond_proj = DiffConv1d(
                in_channels=condition_channels,
                out_channels=model_channels * channel_mult[-1],
                kernel=1,
            )
        elif condition_location == "front":
            self.in_channels = in_channels + condition_channels

        # ---------------- Encoder (flattened, wrapped) ----------------
        self.enc_layers = torch.nn.ModuleList()
        cout = self.in_channels
        self._skip_channels = []

        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level

            if level == 0:
                cin = cout
                cout = model_channels * mult
                conv = DiffConv1d(in_channels=cin, out_channels=cout, kernel=3, **init)
                self.enc_layers.append(WrappedLayer(conv, is_block=False))
                self._skip_channels.append(cout)
            else:
                blk = DiffUNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    down=True,
                    **block_kwargs,
                )
                self.enc_layers.append(WrappedLayer(blk, is_block=True))
                self._skip_channels.append(cout)

            for _ in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                blk = DiffUNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
                self.enc_layers.append(WrappedLayer(blk, is_block=True))
                self._skip_channels.append(cout)

        # ---------------- Decoder (flattened, wrapped) ----------------
        self.dec_layers = torch.nn.ModuleList()
        skip_channels = list(self._skip_channels)

        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level

            if level == len(channel_mult) - 1:
                bottleneck_channels = cout
                if condition_location == "middle":
                    bottleneck_channels += model_channels * channel_mult[-1]

                blk = DiffUNetBlock(
                    in_channels=bottleneck_channels,
                    out_channels=cout,
                    attention=True,
                    **block_kwargs,
                )
                self.dec_layers.append(WrappedLayer(blk, is_block=True))

                blk = DiffUNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    **block_kwargs,
                )
                self.dec_layers.append(WrappedLayer(blk, is_block=True))
            else:
                blk = DiffUNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    up=True,
                    **block_kwargs,
                )
                self.dec_layers.append(WrappedLayer(blk, is_block=True))

            for _ in range(num_blocks + 1):
                cin = cout + skip_channels.pop()
                cout = model_channels * mult
                blk = DiffUNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
                self.dec_layers.append(WrappedLayer(blk, is_block=True))

        self.out_norm = get_group_norm(num_channels=cout)
        temp_out_conv = DiffConv1d(
            in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
        )
        self.out_conv = WrappedLayer(temp_out_conv, is_block=False)
        

    def forward(self, x, cond, noise_labels, class_labels, augment_labels=None):
        # ------------- conditioning -------------
        if self.condition_location == "front":
            x = torch.cat([x, cond], dim=1)

        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)

        if cond is not None and self.condition_location == "embedding":
            emb = emb + self.map_cond(cond)

        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device)
                    >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp)

        # ------------- Encoder -------------
        skips = []
        for layer in self.enc_layers:
            x = layer(x, emb)
            skips.append(x)

        # ------------- Decoder -------------
        for layer in self.dec_layers:
            # access inner block to check in_channels
            inner = layer.layer
            if x.shape[1] != inner.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, emb)

        x = self.out_conv(silu(self.out_norm(x)), emb)
        return x

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import modulus
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import xarray as xr
import h5py

from climsim_utils.data_utils import *
from climsim_datasets import TrainingDataset, ValidationDataset
from unet import Unet
from joint_model import JointModel


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid_info = xr.open_dataset(cfg.grid_info_path)
    input_mean = xr.open_dataset(cfg.input_mean_path)
    input_max = xr.open_dataset(cfg.input_max_path)
    input_min = xr.open_dataset(cfg.input_min_path)
    output_scale = xr.open_dataset(cfg.output_scale_path)
    qn_lbd = np.loadtxt(cfg.qn_lbd_path, delimiter=',')

    data = data_utils(grid_info=grid_info, input_mean=input_mean,
                      input_max=input_max, input_min=input_min,
                      output_scale=output_scale)

    if cfg.variable_subsets == 'v2_rh_mc':
        data.set_to_v2_rh_mc_vars()
    elif cfg.variable_subsets == 'v2_rh':
        data.set_to_v2_rh_vars()
    elif cfg.variable_subsets == 'v2':
        data.set_to_v2_vars()
    elif cfg.variable_subsets == 'v1':
        data.set_to_v1_vars()
    else:
        raise ValueError(f'Unknown variable subset: {cfg.variable_subsets}')

    input_sub, input_div, out_scale = data.save_norm(write=False)

    attn_resolutions = OmegaConf.to_container(cfg.deterministic_model.attn_resolutions, resolve=True)
    channel_mult = OmegaConf.to_container(cfg.deterministic_model.channel_mult, resolve=True)
    resample_filter = OmegaConf.to_container(cfg.deterministic_model.resample_filter, resolve=True)

    model = Unet(
        input_profile_num=data.input_profile_num,
        input_scalar_num=data.input_scalar_num,
        target_profile_num=data.target_profile_num,
        target_scalar_num=data.target_scalar_num,
        output_prune=cfg.output_prune,
        strato_lev_out=cfg.strato_lev_out,
        dropout=cfg.deterministic_model.dropout,
        loc_embedding=cfg.deterministic_model.loc_embedding,
        embedding_type=cfg.deterministic_model.embedding_type,
        num_blocks=cfg.deterministic_model.num_blocks,
        attn_resolutions=attn_resolutions,
        model_channels=cfg.deterministic_model.model_channels,
        skip_conv=cfg.deterministic_model.skip_conv,
        prev_2d=cfg.deterministic_model.prev_2d,
        seq_resolution=cfg.deterministic_model.seq_resolution,
        label_dim=cfg.deterministic_model.label_dim,
        augment_dim=cfg.deterministic_model.augment_dim,
        channel_mult=channel_mult,
        channel_mult_emb=cfg.deterministic_model.channel_mult_emb,
        label_dropout=cfg.deterministic_model.label_dropout,
        channel_mult_noise=cfg.deterministic_model.channel_mult_noise,
        encoder_type=cfg.deterministic_model.encoder_type,
        decoder_type=cfg.deterministic_model.decoder_type,
        resample_filter=resample_filter,
    ).to(device)

    print(f"Loading checkpoint from {cfg.restart_path}")
    model_ckpt = modulus.Module.from_checkpoint(cfg.restart_path).to(device)
    model.load_state_dict(model_ckpt.state_dict())
    model.eval()

    # We need reshape helpers from JointModel — instantiate a minimal wrapper
    # just for its reshape methods (res_model and buffers are unused here)
    res_std = torch.load(cfg.res_std_path).to(torch.float32)
    res_mean = torch.load(cfg.res_mean_path).to(torch.float32)
    preds_std = torch.load(cfg.preds_std_path).to(torch.float32)
    preds_mean = torch.load(cfg.preds_mean_path).to(torch.float32)

    from physicsnemo.models.diffusion.preconditioning import EDMPrecond
    res_attn_resolutions = OmegaConf.to_container(cfg.diffusion_model.attn_resolutions, resolve=True)
    res_channel_mult = OmegaConf.to_container(cfg.diffusion_model.channel_mult, resolve=True)
    cond_channels = (data.target_profile_num + data.target_scalar_num +
                     data.input_profile_num + data.input_scalar_num) * 64
    res_model = EDMPrecond(
        img_resolution=cfg.diffusion_model.seq_resolution,
        img_channels=data.target_profile_num + data.target_scalar_num,
        input_profile_num=data.target_profile_num,
        input_scalar_num=data.target_scalar_num,
        use_fp16=False,
        model_type="DhariwalUNet",
        img_in_channels=data.target_profile_num + data.target_scalar_num,
        img_out_channels=data.target_profile_num + data.target_scalar_num,
        attn_resolutions=res_attn_resolutions,
        num_blocks=cfg.diffusion_model.num_blocks,
        model_channels=cfg.diffusion_model.model_channels,
        channel_mult=res_channel_mult,
        condition=cfg.diffusion_model.condition,
        condition_channels=cond_channels,
        condition_location=cfg.diffusion_model.condition_location,
    ).to(device)

    joint_model = JointModel(
        deterministic_model=model,
        res_model=res_model,
        res_std=res_std, res_mean=res_mean,
        preds_std=preds_std, preds_mean=preds_mean,
        input_profile_num=data.input_profile_num,
        input_scalar_num=data.input_scalar_num,
        target_profile_num=data.target_profile_num,
        target_scalar_num=data.target_scalar_num,
        condition_channel_num=cond_channels,
        condition_type=cfg.diffusion_model.condition_type,
        condtition_location=cfg.diffusion_model.condition_location,
    ).to(device)

    def run_and_save(dataset, preds_path, targets_path, desc):
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=cfg.num_workers,
        )
        n = len(dataset)
        row = 0
        preds_ds = targets_ds = None
        with h5py.File(preds_path, 'w') as pf, h5py.File(targets_path, 'w') as tf:
            with torch.no_grad():
                for x, y in tqdm(loader, desc=desc):
                    x = x.to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                        inp = joint_model.reshape_input(x)
                        out, _ = joint_model.deterministic_model(inp)
                        out = joint_model.reverse_reshape_target(out)
                    out_np = out.float().cpu().numpy()
                    y_np = y.numpy()
                    b = out_np.shape[0]

                    if preds_ds is None:
                        preds_ds = pf.create_dataset('data', shape=(n, out_np.shape[1]), dtype=np.float32)
                        targets_ds = tf.create_dataset('data', shape=(n, y_np.shape[1]), dtype=np.float32)

                    preds_ds[row:row + b] = out_np
                    targets_ds[row:row + b] = y_np
                    row += b

        print(f"Saved {row} samples -> {preds_path}")

    train_dataset = TrainingDataset(
        parent_path=cfg.data_path,
        input_sub=input_sub, input_div=input_div, out_scale=out_scale,
        qinput_prune=cfg.qinput_prune, output_prune=cfg.output_prune,
        strato_lev=cfg.strato_lev, qn_lbd=qn_lbd,
        decouple_cloud=cfg.decouple_cloud, aggressive_pruning=cfg.aggressive_pruning,
        strato_lev_qc=cfg.strato_lev_qc, strato_lev_qinput=cfg.strato_lev_qinput,
        strato_lev_tinput=cfg.strato_lev_tinput, strato_lev_out=cfg.strato_lev_out,
        input_clip=cfg.input_clip, input_clip_rhonly=cfg.input_clip_rhonly,
    )

    val_dataset = ValidationDataset(
        val_input_path=cfg.val_input_path, val_target_path=cfg.val_target_path,
        input_sub=input_sub, input_div=input_div, out_scale=out_scale,
        qinput_prune=cfg.qinput_prune, output_prune=cfg.output_prune,
        strato_lev=cfg.strato_lev, qn_lbd=qn_lbd,
        decouple_cloud=cfg.decouple_cloud, aggressive_pruning=cfg.aggressive_pruning,
        strato_lev_qc=cfg.strato_lev_qc, strato_lev_qinput=cfg.strato_lev_qinput,
        strato_lev_tinput=cfg.strato_lev_tinput, strato_lev_out=cfg.strato_lev_out,
        input_clip=cfg.input_clip, input_clip_rhonly=cfg.input_clip_rhonly,
    )

    run_and_save(train_dataset, cfg.train_preds_path, cfg.train_targets_path,
                 desc="Train preds")
    run_and_save(val_dataset, cfg.val_preds_path, cfg.val_targets_path,
                 desc="Val preds")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import argparse
import h5py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os, string, sys
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import torch
from tqdm import tqdm

from climsim_utils.data_utils import *
from omegaconf import OmegaConf
from modulus import Module
from baseline_models.unet.training_default.joint_model import JointModel

parser = argparse.ArgumentParser()
parser.add_argument('--n_batches',            type=int, default=None)
parser.add_argument('--diff_config_path',     type=str, default=None,
                    help='Path to saved_config.yaml for the joint diffusion model.')
parser.add_argument('--diff_checkpoint_path', type=str, default='',
                    help='Optional .mdlus checkpoint path (overrides diff_model.pt in config).')
parser.add_argument('--diff_n_batches',       type=int, default=None,
                    help='Limit number of timestep batches for diffusion inference.')
parser.add_argument('--diff_start_doy',       type=int, default=None,
                    help='Start day-of-year (0-indexed) for the diffusion npy data '
                         '(defaults to START_DOY used by the h5 data).')
args = parser.parse_args()

# --- Paths ---
grid_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc'

input_mean_v2_rh_mc_file   = 'input_mean_v2_rh_mc_pervar.nc'
input_max_v2_rh_mc_file    = 'input_max_v2_rh_mc_pervar.nc'
input_min_v2_rh_mc_file    = 'input_min_v2_rh_mc_pervar.nc'
output_scale_v2_rh_mc_file = 'output_scale_std_lowerthred_v2_rh_mc.nc'

preds_path   = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/val_preds.h5'
targets_path = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/val_targets.h5'

input_npy_path  = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/val_set/val_input.npy'
target_npy_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/val_set/val_target.npy'

save_path = '/global/homes/k/kfrields/climsim-kaggle-edition/figures/offline/spatial_variance'

# --- Grid and normalizations ---
print('Opening grid info...', flush=True)
grid_info = xr.open_dataset(grid_path)
grid_area = grid_info['area'].values
level     = grid_info.lev.values
lat       = grid_info['lat'].values
lon       = ((grid_info['lon'].values + 180) % 360) - 180

print('Opening normalizations...', flush=True)
input_mean_v2_rh_mc   = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/'  + input_mean_v2_rh_mc_file)
input_max_v2_rh_mc    = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/'  + input_max_v2_rh_mc_file)
input_min_v2_rh_mc    = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/'  + input_min_v2_rh_mc_file)
output_scale_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/outputs/' + output_scale_v2_rh_mc_file)

print('Initializing data_utils...', flush=True)
data_utils_obj = data_utils(
    grid_info    = grid_info,
    input_mean   = input_mean_v2_rh_mc,
    input_max    = input_max_v2_rh_mc,
    input_min    = input_min_v2_rh_mc,
    output_scale = output_scale_v2_rh_mc,
    qinput_log   = False,
    normalize    = False,
)
data_utils_obj.set_to_v2_rh_mc_vars()

lat_bin_mids = data_utils_obj.lat_bin_mids
ncol         = grid_area.shape[0]

# Precompute column → latitude-bin assignments (uniform bin spacing assumed)
lat_bin_mids_arr = np.asarray(lat_bin_mids)
n_lat_bins       = len(lat_bin_mids_arr)
half_step        = (lat_bin_mids_arr[1] - lat_bin_mids_arr[0]) / 2.0
lat_bin_edges    = np.concatenate([
    [lat_bin_mids_arr[0]  - half_step],
    (lat_bin_mids_arr[:-1] + lat_bin_mids_arr[1:]) / 2,
    [lat_bin_mids_arr[-1] + half_step],
])
col_bin = np.clip(np.digitize(lat, lat_bin_edges) - 1, 0, n_lat_bins - 1)  # (ncol,)

# --- Variable settings ---
vars_list = ['DTPHYS', 'DQ1PHYS', 'DQnPHYS', 'DUPHYS', 'DVPHYS']

def get_var_settings(var):
    if var == 'DTPHYS':
        return 'Heating Tendency',           'K/s',    0
    elif var == 'DQ1PHYS':
        return 'RH Tendency',                '',       60
    elif var == 'DQnPHYS':
        return 'Liquid+Ice Cloud Tendency',  'mg/kg/s', 120
    elif var == 'DUPHYS':
        return 'Zonal Wind Tendency',        'm/s²',   180
    elif var == 'DVPHYS':
        return 'Meridional Wind Tendency',   'm/s²',   240

n_levels = 60

latitude_ticks  = [-60, -30, 0, 30, 60]
latitude_labels = ['60S', '30S', '0', '30N', '60N']

TIMESTEP_MINUTES = 20
START_DOY        = 31

_MONTH_DOY_START = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

def _timestep_month(t):
    doy = (START_DOY + t * TIMESTEP_MINUTES // (60 * 24)) % 365
    for m in range(11, -1, -1):
        if doy >= _MONTH_DOY_START[m]:
            return m + 1
    return 1

SEASONS = {
    'DJF': {'label': 'Winter (DJF)', 'months': {12, 1, 2}},
    'MAM': {'label': 'Spring (MAM)', 'months': {3, 4, 5}},
    'JJA': {'label': 'Summer (JJA)', 'months': {6, 7, 8}},
    'SON': {'label': 'Fall (SON)',   'months': {9, 10, 11}},
}


# ---------------------------------------------------------------------------
# Variance helpers
# ---------------------------------------------------------------------------

def zonal_variance_da(arr_3d):
    """Area-weighted zonal spatial variance of (time, ncol, lev) -> DataArray (lev, lat).

    For each time step and latitude bin, computes the area-weighted variance
    across columns within that bin, then averages over time.
    """
    n_t   = arr_3d.shape[0]
    n_lev = arr_3d.shape[2]
    zv    = np.zeros((n_t, n_lat_bins, n_lev), dtype=np.float64)

    for b in range(n_lat_bins):
        mask = col_bin == b
        if not mask.any():
            continue
        cols = arr_3d[:, mask, :]                                             # (n_t, n_cols_in_bin, n_lev)
        w    = grid_area[mask] / grid_area[mask].sum()                        # normalised weights
        wm   = (cols * w[None, :, None]).sum(axis=1, keepdims=True)           # weighted mean
        zv[:, b, :] = (w[None, :, None] * (cols - wm) ** 2).sum(axis=1)      # weighted variance

    return xr.DataArray(
        zv.mean(axis=0).T,                                                    # (n_lev, n_lat_bins)
        dims=['hybrid pressure (hPa)', 'latitude'],
        coords={'hybrid pressure (hPa)': level, 'latitude': lat_bin_mids},
    )


def global_variance_profile(arr_3d):
    """Area-weighted global spatial variance of (time, ncol, lev) -> ndarray (lev,).

    Variance across all ncol columns at each time step, averaged over time.
    """
    w  = grid_area / grid_area.sum()                                          # (ncol,)
    wm = (arr_3d * w[None, :, None]).sum(axis=1, keepdims=True)              # (n_t, 1, n_lev)
    wv = (w[None, :, None] * (arr_3d - wm) ** 2).sum(axis=1)                 # (n_t, n_lev)
    return wv.mean(axis=0)                                                     # (n_lev,)


def compute_zonal_variance(var, preds_ds, targets_ds, n_rows, n_time, time_mask=None):
    """Return (true_da, pred_da) area-weighted zonal variance DataArrays for `var`."""
    _, _, var_index = get_var_settings(var)
    sl = slice(var_index, var_index + n_levels)

    pred_var   = preds_ds  [:n_rows, sl].reshape(n_time, ncol, n_levels)
    target_var = targets_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)

    if time_mask is not None:
        pred_var   = pred_var  [time_mask]
        target_var = target_var[time_mask]

    true_da = zonal_variance_da(target_var)
    pred_da = zonal_variance_da(pred_var)
    del pred_var, target_var
    return true_da, pred_da


def compute_global_variance(var, preds_ds, targets_ds, n_rows, n_time, time_mask=None):
    """Return (true_prof, pred_prof) global spatial variance profiles, shape (n_levels,)."""
    _, _, var_index = get_var_settings(var)
    sl = slice(var_index, var_index + n_levels)

    pred_var   = preds_ds  [:n_rows, sl].reshape(n_time, ncol, n_levels)
    target_var = targets_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)

    if time_mask is not None:
        pred_var   = pred_var  [time_mask]
        target_var = target_var[time_mask]

    true_prof = global_variance_profile(target_var)
    pred_prof = global_variance_profile(pred_var)
    del pred_var, target_var
    return true_prof, pred_prof


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_all_spatial_variances(preds_ds, targets_ds, n_rows, n_time, time_mask=None,
                                title='True vs Predicted Spatial Variance',
                                fname='spatial_variance_all.png', show=True, out_dir=None):
    """Rows = variables; 3 cols = (true variance | predicted variance | ratio pred/true)."""
    n_vars = len(vars_list)
    labels = [f'({l})' for l in string.ascii_lowercase[:n_vars * 3]]

    fig, axs = plt.subplots(n_vars, 3, figsize=(15, 2.8 * n_vars), constrained_layout=True)
    if n_vars == 1:
        axs = axs[np.newaxis, :]

    for row, var in enumerate(vars_list):
        var_title, unit, _ = get_var_settings(var)
        true_da, pred_da   = compute_zonal_variance(var, preds_ds, targets_ds,
                                                     n_rows, n_time, time_mask)
        ratio_da = pred_da / true_da.where(true_da != 0)

        vmax_var = float(np.nanpercentile(true_da.values, 99))

        cols = [
            (true_da,  'True Variance',       'viridis', 0,   vmax_var),
            (pred_da,  'Predicted Variance',   'viridis', 0,   vmax_var),
            (ratio_da, 'Ratio (Pred / True)',  'RdBu_r',  0.5, 1.5),
        ]
        for col, (da, col_title, cmap, c_vmin, c_vmax) in enumerate(cols):
            ax = axs[row, col]
            im = da.plot(ax=ax, add_colorbar=False, cmap=cmap, vmin=c_vmin, vmax=c_vmax)
            cbar_label = f'{unit}²' if col < 2 else '(dimensionless)'
            fig.colorbar(im, ax=ax, label=cbar_label, pad=0.02)
            ax.set_title(f"{labels[row*3+col]} {var_title} — {col_title}", fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Latitude', fontsize=7)
            ax.set_ylabel('Hybrid pressure (hPa)' if col == 0 else '', fontsize=7)
            ax.set_xticks(latitude_ticks)
            ax.set_xticklabels(latitude_labels, fontsize=7)
            ax.tick_params(axis='y', labelsize=7)

    fig.suptitle(title, fontsize=11)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=200, bbox_inches='tight')
        print(f'Saved: {fpath}', flush=True)
    if show:
        plt.show()
    else:
        plt.close()


def plot_global_variance_profiles(preds_ds, targets_ds, n_rows, n_time, time_mask=None,
                                   title='Global Spatial Variance Profile',
                                   fname='global_variance_profiles.png', show=True, out_dir=None):
    """One panel per variable: true vs predicted global spatial variance vs pressure level."""
    n_vars = len(vars_list)
    labels = [f'({l})' for l in string.ascii_lowercase[:n_vars]]

    fig, axs = plt.subplots(1, n_vars, figsize=(3.5 * n_vars, 5), constrained_layout=True)
    if n_vars == 1:
        axs = [axs]

    for col, var in enumerate(vars_list):
        var_title, unit, _ = get_var_settings(var)
        true_prof, pred_prof = compute_global_variance(var, preds_ds, targets_ds,
                                                        n_rows, n_time, time_mask)
        ax = axs[col]
        ax.plot(true_prof, level, label='True',      color='black',    linewidth=1.5)
        ax.plot(pred_prof, level, label='Predicted', color='tab:blue', linewidth=1.5, linestyle='--')
        ax.invert_yaxis()
        ax.set_xlabel(f'Spatial Variance [{unit}²]', fontsize=8)
        ax.set_ylabel('Hybrid pressure (hPa)' if col == 0 else '', fontsize=8)
        ax.set_title(f'{labels[col]} {var_title}', fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=11)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=200, bbox_inches='tight')
        print(f'Saved: {fpath}', flush=True)
    if show:
        plt.show()
    else:
        plt.close()


def plot_column_variance_map(targets, det_preds, diff_preds, n_rows, n_time, time_mask=None,
                              title='Column Residual Variance',
                              fname='column_residual_variance_map.png', show=True, out_dir=None):
    """Rows = variables; 3 cols = Robinson maps of:
       (true residual variance | predicted residual variance | ratio pred/true).

    True residual  = target - det_pred    (what the diffusion model should predict)
    Pred residual  = diff_pred            (what it actually predicted)
    Variance is computed over the time dimension then averaged over pressure levels,
    giving one scalar per (ncol,) grid point.
    """
    n_vars       = len(vars_list)
    panel_labels = [f'({l})' for l in string.ascii_lowercase[:n_vars * 3]]

    fig, axs = plt.subplots(
        n_vars, 3,
        figsize=(24, 3.5 * n_vars),
        subplot_kw={'projection': ccrs.Robinson()},
        constrained_layout=True,
    )
    if n_vars == 1:
        axs = axs[np.newaxis, :]

    for row, var in enumerate(vars_list):
        var_title, unit, var_index = get_var_settings(var)
        sl = slice(var_index, var_index + n_levels)

        targ_var = targets   [:n_rows, sl].reshape(n_time, ncol, n_levels)
        det_var  = det_preds [:n_rows, sl].reshape(n_time, ncol, n_levels)
        dif_var  = diff_preds[:n_rows, sl].reshape(n_time, ncol, n_levels)
        if time_mask is not None:
            targ_var = targ_var[time_mask]
            det_var  = det_var [time_mask]
            dif_var  = dif_var [time_mask]

        # variance over time, averaged over levels → (ncol,)
        true_map  = (targ_var - det_var).var(axis=0).mean(axis=1)
        pred_map  = dif_var.var(axis=0).mean(axis=1)
        ratio_map = np.where(true_map > 0, pred_map / true_map, np.nan)
        del targ_var, det_var, dif_var

        vmax_var = float(np.nanpercentile(true_map, 99))
        cols = [
            (true_map,  'Var(True Residual)',  'viridis', 0,   vmax_var),
            (pred_map,  'Var(Pred Residual)',   'viridis', 0,   vmax_var),
            (ratio_map, 'Ratio (Pred / True)',  'RdBu_r',  0.5, 1.5),
        ]
        for col, (data, col_title, cmap, c_vmin, c_vmax) in enumerate(cols):
            ax = axs[row, col]
            levels_c = np.linspace(c_vmin, c_vmax, 20)
            tc = ax.tricontourf(lon, lat, data, transform=ccrs.PlateCarree(),
                                cmap=cmap, levels=levels_c, extend='both',
                                vmin=c_vmin, vmax=c_vmax)
            ax.coastlines(linewidth=0.5, color='black')
            ax.set_global()
            ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            ax.set_title(f"{panel_labels[row*3+col]} {var_title} — {col_title}", fontsize=9)
            cbar_label = f'{unit}²' if col < 2 else '(dimensionless)'
            cbar = fig.colorbar(tc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label(cbar_label, fontsize=9)
            cbar.locator = ticker.MaxNLocator(nbins=4)

    fig.suptitle(title, fontsize=11)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=200, bbox_inches='tight')
        print(f'Saved: {fpath}', flush=True)
    if show:
        plt.show()
    else:
        plt.close()


def plot_seasonal_spatial_variances(preds_ds, targets_ds, n_rows, n_time,
                                     season_masks, out_dir=None):
    """Zonal variance, global profile, and column map plots, one set per season."""
    for key, info in SEASONS.items():
        mask = season_masks[key]
        if len(mask) == 0:
            print(f'  No timesteps for {key}, skipping.', flush=True)
            continue
        print(f'  {info["label"]} ({len(mask)} timesteps)...', flush=True)
        plot_all_spatial_variances(
            preds_ds, targets_ds, n_rows, n_time,
            time_mask = mask,
            title     = f'True vs Predicted Spatial Variance — {info["label"]}',
            fname     = f'spatial_variance_{key.lower()}.png',
            show      = False, out_dir = out_dir,
        )
        plot_global_variance_profiles(
            preds_ds, targets_ds, n_rows, n_time,
            time_mask = mask,
            title     = f'Global Spatial Variance — {info["label"]}',
            fname     = f'global_variance_profiles_{key.lower()}.png',
            show      = False, out_dir = out_dir,
        )


# ---------------------------------------------------------------------------
# Diffusion model loading and inference (mirrors plot_residual_zonal_means.py)
# ---------------------------------------------------------------------------

def load_joint_model(config_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config_path) as f:
        cfg = OmegaConf.load(f)

    gi   = xr.open_dataset(cfg.grid_info_path)
    imn  = xr.open_dataset(cfg.input_mean_path)
    imx  = xr.open_dataset(cfg.input_max_path)
    imi  = xr.open_dataset(cfg.input_min_path)
    osc  = xr.open_dataset(cfg.output_scale_path)
    qn_lbd = np.loadtxt(cfg.qn_lbd_path, delimiter=',')

    res_std    = torch.load(cfg.res_std_path,    map_location=device).to(torch.float32)
    res_mean   = torch.load(cfg.res_mean_path,   map_location=device).to(torch.float32)
    preds_std  = torch.load(cfg.preds_std_path,  map_location=device).to(torch.float32)
    preds_mean = torch.load(cfg.preds_mean_path, map_location=device).to(torch.float32)

    diff_data = data_utils(
        grid_info=gi, input_mean=imn, input_max=imx, input_min=imi,
        output_scale=osc, qinput_log=False, normalize=False,
        res_std=res_std, res_mean=res_mean,
        preds_std=preds_std, preds_mean=preds_mean,
    )
    diff_data.set_to_v2_rh_mc_vars()

    input_sub, input_div, out_scale = diff_data.save_norm(write=False)
    input_sub    = input_sub[None, :]
    input_div    = input_div[None, :]
    out_scale_np = out_scale[None, :]

    print(f'  np.load input  ({input_npy_path})...', flush=True)
    npy_input  = np.load(input_npy_path)
    print(f'  np.load target ({target_npy_path})...', flush=True)
    npy_target = np.load(target_npy_path)
    print(f'  Preprocessing input array (shape {npy_input.shape})...', flush=True)

    npy_input[:, 120:180] = 1 - np.exp(-npy_input[:, 120:180] * qn_lbd)
    npy_input = (npy_input - input_sub) / input_div
    npy_input = np.where(np.isnan(npy_input), 0, npy_input)
    npy_input = np.where(np.isinf(npy_input), 0, npy_input)
    npy_input[:, 120:135] = 0
    npy_input[:, 60:120]  = np.clip(npy_input[:, 60:120], 0, 1.2)
    print('  Converting to torch tensor...', flush=True)
    torch_input = torch.tensor(npy_input).float()

    ncol_diff   = diff_data.num_latlon
    n_time_diff = npy_input.shape[0] // ncol_diff
    targets_flat = npy_target[:n_time_diff * ncol_diff]

    base = os.path.join(cfg.save_path, cfg.expname)
    print(f'  Loading deterministic model ({os.path.join(base, "unet_model.pt")})...', flush=True)
    det_model = torch.jit.load(os.path.join(base, 'unet_model.pt')).to(device)
    det_model.eval()

    if checkpoint_path:
        print(f'  Loading diffusion model from checkpoint ({checkpoint_path})...', flush=True)
        diff_model = Module.from_checkpoint(checkpoint_path).to(device)
    else:
        print(f'  Loading diffusion model ({os.path.join(base, "diff_model.pt")})...', flush=True)
        diff_model = torch.load(os.path.join(base, 'diff_model.pt')).to(device)
    diff_model.eval()
    print('  Models loaded.', flush=True)

    loc   = cfg.diffusion_model.condition_location
    ctype = cfg.diffusion_model.condition_type
    base_ch = (diff_data.target_profile_num + diff_data.target_scalar_num +
               diff_data.input_profile_num  + diff_data.input_scalar_num)
    if loc == 'front':
        cond_channels = base_ch
    if loc == 'embedding':
        cond_channels = base_ch * 64 if ctype == 'input_output' else 8192
    elif loc in ('middle', 'cross'):
        cond_channels = base_ch

    joint_model = JointModel(
        det_model, diff_model, res_std, res_mean, preds_std, preds_mean,
        input_profile_num      = diff_data.input_profile_num,
        input_scalar_num       = diff_data.input_scalar_num,
        target_profile_num     = diff_data.target_profile_num,
        target_scalar_num      = diff_data.target_scalar_num,
        condition_channel_num  = cond_channels,
        condition_type         = ctype,
        condtition_location    = loc,
        p_mean                 = cfg.diffusion_model.p_mean,
        p_std                  = cfg.diffusion_model.p_std,
        t_sampling             = cfg.diffusion_model.t_sampling,
    ).to(device)

    return joint_model, torch_input, targets_flat, n_time_diff, diff_data, out_scale_np


def run_diffusion_inference(joint_model, diff_data, torch_input, out_scale_np,
                             n_batches_limit=None,
                             sigma_min=0.1, sigma_max=30, num_steps=18, rho=7):
    device     = next(joint_model.parameters()).device
    batch_size = diff_data.num_latlon
    joint_model.eval()

    det_list, joint_list, diff_list = [], [], []
    n_done = 0

    with torch.no_grad():
        for i in tqdm(range(0, torch_input.shape[0], batch_size), desc='Diffusion inference'):
            if n_batches_limit is not None and n_done >= n_batches_limit:
                break

            input_batch = torch_input[i : i + batch_size].to(device)
            input_batch = joint_model.reshape_input(input_batch)

            output, _ = joint_model.deterministic_model(input_batch)

            safe_std  = torch.clamp(joint_model.res_std, min=1e-2)
            cond_out  = ((output - joint_model.preds_mean) /
                         (joint_model.preds_std + 1e-8)) * 0.5
            loc = joint_model.condition_location
            if loc == 'front' and joint_model.condition_type == 'input_output':
                condition_data = torch.cat((input_batch, cond_out), dim=1)
            if loc == 'embedding' and joint_model.condition_type == 'input_output':
                lc = torch.cat((input_batch, cond_out), dim=1)
                condition_data = lc.reshape(lc.shape[0], -1)
            elif loc in ('middle', 'cross') and joint_model.condition_type == 'input_output':
                condition_data = torch.cat((input_batch, cond_out), dim=1)

            latents = torch.randn(
                (batch_size, diff_data.target_profile_num + diff_data.target_scalar_num, 64),
                device=device)
            res = joint_model.res_model.edm_sampler(
                latents, condition_input=condition_data,
                sigma_min=sigma_min, sigma_max=sigma_max,
                rho=rho, num_steps=num_steps)

            denorm_res   = (res / 0.5) * (safe_std + 1e-8)
            reshaped_res = joint_model.reverse_reshape_target(denorm_res)
            reshaped_out = joint_model.reverse_reshape_target(output)

            joint_pred = reshaped_out + reshaped_res
            joint_pred[:, 300:] = torch.nn.functional.relu(joint_pred[:, 300:])

            det_list.append(  (reshaped_out.cpu().numpy() / out_scale_np))
            joint_list.append((joint_pred.cpu().numpy()   / out_scale_np))
            diff_list.append( (reshaped_res.cpu().numpy() / out_scale_np))
            n_done += 1

    return (np.concatenate(joint_list, axis=0),
            np.concatenate(det_list,   axis=0),
            np.concatenate(diff_list,  axis=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

n_batches_to_load = args.n_batches

# ---------------------------------------------------------------------------
# Optional: diffusion model inference + variance visualisations
# ---------------------------------------------------------------------------
if args.diff_config_path:

    print('\n=== Diffusion model ===', flush=True)
    print('Loading joint model...', flush=True)
    joint_model, torch_input_diff, targets_flat_diff, n_time_diff, diff_data, out_scale_diff = \
        load_joint_model(
            config_path     = args.diff_config_path,
            checkpoint_path = args.diff_checkpoint_path,
        )

    print('Running inference...', flush=True)
    joint_preds_flat, det_preds_flat, diff_preds_flat = run_diffusion_inference(
        joint_model, diff_data, torch_input_diff, out_scale_diff,
        n_batches_limit = args.diff_n_batches,
    )

    ncol_diff   = diff_data.num_latlon
    n_time_used = joint_preds_flat.shape[0] // ncol_diff
    n_rows_diff = n_time_used * ncol_diff

    diff_start_doy = args.diff_start_doy if args.diff_start_doy is not None else START_DOY

    def _diff_timestep_month(t):
        doy = (diff_start_doy + t * TIMESTEP_MINUTES // (60 * 24)) % 365
        for m in range(11, -1, -1):
            if doy >= _MONTH_DOY_START[m]:
                return m + 1
        return 1

    diff_months = np.array([_diff_timestep_month(t) for t in range(n_time_used)])
    diff_season_masks = {
        key: np.where(np.isin(diff_months, list(info['months'])))[0]
        for key, info in SEASONS.items()
    }

    diff_save_path = os.path.join(save_path, 'diffusion')

    print('Plotting annual column residual variance map...', flush=True)
    plot_column_variance_map(
        targets_flat_diff, det_preds_flat, diff_preds_flat, n_rows_diff, n_time_used,
        title   = 'Column Residual Variance — Annual',
        fname   = 'column_residual_variance_map_annual.png',
        show    = False, out_dir = diff_save_path,
    )

    print('Plotting seasonal column residual variance maps...', flush=True)
    for key, info in SEASONS.items():
        mask = diff_season_masks[key]
        if len(mask) == 0:
            print(f'  No timesteps for {key}, skipping.', flush=True)
            continue
        print(f'  {info["label"]} ({len(mask)} timesteps)...', flush=True)
        plot_column_variance_map(
            targets_flat_diff, det_preds_flat, diff_preds_flat, n_rows_diff, n_time_used,
            time_mask = mask,
            title     = f'Column Residual Variance — {info["label"]}',
            fname     = f'column_residual_variance_map_{key.lower()}.png',
            show      = False, out_dir = diff_save_path,
        )

    print('=== Diffusion model done ===\n', flush=True)

print('Opening h5 files...', flush=True)
with h5py.File(preds_path, 'r') as preds_f, h5py.File(targets_path, 'r') as targets_f:
    preds_ds   = preds_f['data']
    targets_ds = targets_f['data']

    n_output_vars = preds_ds.shape[1]
    n_time_total  = preds_ds.shape[0] // ncol
    n_time        = min(n_batches_to_load, n_time_total) if n_batches_to_load else n_time_total
    n_rows        = n_time * ncol
    print(f'ncol={ncol}, n_time={n_time}/{n_time_total}, n_output_vars={n_output_vars}', flush=True)

    months = np.array([_timestep_month(t) for t in range(n_time)])
    season_masks = {
        key: np.where(np.isin(months, list(info['months'])))[0]
        for key, info in SEASONS.items()
    }

    print('Plotting annual zonal spatial variances...', flush=True)
    plot_all_spatial_variances(preds_ds, targets_ds, n_rows, n_time,
                                show=False, out_dir=save_path)

    print('Plotting annual global variance profiles...', flush=True)
    plot_global_variance_profiles(preds_ds, targets_ds, n_rows, n_time,
                                   show=False, out_dir=save_path)

    '''print('Plotting seasonal spatial variances...', flush=True)
    plot_seasonal_spatial_variances(preds_ds, targets_ds, n_rows, n_time, season_masks,
                                     out_dir=save_path)'''

print('All done.', flush=True)

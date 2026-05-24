# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os, string
from climsim_utils.data_utils import *

# --- Paths ---
grid_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc'

input_mean_v2_rh_mc_file = 'input_mean_v2_rh_mc_pervar.nc'
input_max_v2_rh_mc_file  = 'input_max_v2_rh_mc_pervar.nc'
input_min_v2_rh_mc_file  = 'input_min_v2_rh_mc_pervar.nc'
output_scale_v2_rh_mc_file = 'output_scale_std_lowerthred_v2_rh_mc.nc'

preds_path   = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/train_preds.npy'
targets_path = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/train_targets.npy'

save_path = '/global/homes/k/kfrields/climsim-kaggle-edition/figures/offline/residual_zonal_means'

# --- Grid and normalizations ---
grid_info = xr.open_dataset(grid_path)
grid_area = grid_info['area'].values
level     = grid_info.lev.values

input_mean_v2_rh_mc  = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/'  + input_mean_v2_rh_mc_file)
input_max_v2_rh_mc   = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/'  + input_max_v2_rh_mc_file)
input_min_v2_rh_mc   = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/'  + input_min_v2_rh_mc_file)
output_scale_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/outputs/' + output_scale_v2_rh_mc_file)

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

# --- Variable settings ---
# Each 3D output variable has 60 pressure levels; var_index is the first column in the flat output array.
var_settings = {
    'DTPHYS':  {'var_title': 'Heating Tendency',         'scaling': 1.,   'unit': 'K/s',      'var_index': 0,   'vmax': 5e-7,  'vmin': -5e-7},
    'DQ1PHYS': {'var_title': 'Moistening Tendency',      'scaling': 1e3,  'unit': 'g/kg/s',   'var_index': 60,  'vmax': 1e-6,  'vmin': -1e-6},
    'DQ2PHYS': {'var_title': 'Liquid Cloud Tendency',    'scaling': 1e6,  'unit': 'mg/kg/s',  'var_index': 120, 'vmax': 1e-3,  'vmin': -1e-3},
    'DQ3PHYS': {'var_title': 'Ice Cloud Tendency',       'scaling': 1e6,  'unit': 'mg/kg/s',  'var_index': 180, 'vmax': 1e-3,  'vmin': -1e-3},
    'DUPHYS':  {'var_title': 'Zonal Wind Tendency',      'scaling': 1.,   'unit': 'm/s²',     'var_index': 240, 'vmax': 5e-7,  'vmin': -5e-7},
    'DVPHYS':  {'var_title': 'Meridional Wind Tendency', 'scaling': 1.,   'unit': 'm/s²',     'var_index': 300, 'vmax': 5e-7,  'vmin': -5e-7},
}

n_levels = 60

latitude_ticks  = [-60, -30, 0, 30, 60]
latitude_labels = ['60S', '30S', '0', '30N', '60N']


def zonal_mean_da(arr_3d):
    """Area-weighted zonal mean of (time, ncol, lev) -> xr.DataArray (lev, lat)."""
    zm  = data_utils_obj.zonal_bin_weight_3d(arr_3d)  # (time, n_lat_bins, lev)
    zm  = zm.mean(axis=0)                              # (n_lat_bins, lev)
    return xr.DataArray(
        zm.T,
        dims=['hybrid pressure (hPa)', 'latitude'],
        coords={'hybrid pressure (hPa)': level, 'latitude': lat_bin_mids},
    )


def plot_residual_zonal_mean(var, preds_3d, targets_3d, show=True, out_dir=None):
    """Two-panel plot: mean bias and MAE zonal mean for one variable."""
    s    = var_settings[var]
    idx  = s['var_index']
    sc   = s['scaling']

    pred_var   = preds_3d[:, :, idx:idx + n_levels]    # (time, ncol, lev)
    target_var = targets_3d[:, :, idx:idx + n_levels]

    residual = target_var - pred_var                   # signed error (target - pred)
    abs_err  = np.abs(residual)

    bias_da = sc * zonal_mean_da(residual)
    mae_da  = sc * zonal_mean_da(abs_err)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = [f'({l})' for l in string.ascii_lowercase[:2]]

    # --- Bias panel ---
    im0 = bias_da.plot(ax=axs[0], add_colorbar=False, cmap='RdBu_r',
                       vmin=s['vmin'], vmax=s['vmax'])
    fig.colorbar(im0, ax=axs[0], label=f"{s['var_title']} ({s['unit']})")
    axs[0].set_title(f"{labels[0]} Mean Bias (Target − Pred)")
    axs[0].invert_yaxis()
    axs[0].set_xlabel('Latitude')
    axs[0].set_ylabel('Hybrid pressure (hPa)')
    axs[0].set_xticks(latitude_ticks)
    axs[0].set_xticklabels(latitude_labels)

    # --- MAE panel ---
    im1 = mae_da.plot(ax=axs[1], add_colorbar=False, cmap='viridis')
    fig.colorbar(im1, ax=axs[1], label=f"{s['var_title']} ({s['unit']})")
    axs[1].set_title(f"{labels[1]} MAE")
    axs[1].invert_yaxis()
    axs[1].set_xlabel('Latitude')
    axs[1].set_ylabel('')
    axs[1].set_xticks(latitude_ticks)
    axs[1].set_xticklabels(latitude_labels)

    plt.suptitle(f"{s['var_title']} — Zonal Mean Residuals", fontsize=13)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f'residual_zonal_mean_{var}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Saved: {fname}')
    if show:
        plt.show()
    else:
        plt.close()


# --- Load precomputed predictions and targets ---
print('Loading predictions...')
preds   = np.load(preds_path)    # (n_samples, n_output_vars)
print('Loading targets...')
targets = np.load(targets_path)  # (n_samples, n_output_vars)

n_samples    = preds.shape[0]
n_output_vars = preds.shape[1]
n_time       = n_samples // ncol

print(f'n_samples={n_samples}, ncol={ncol}, n_time={n_time}, n_output_vars={n_output_vars}')

preds_3d   = preds.reshape(n_time, ncol, n_output_vars)    # (time, ncol, n_vars)
targets_3d = targets.reshape(n_time, ncol, n_output_vars)

# --- Plot ---
for var in var_settings:
    print(f'Plotting {var}...')
    plot_residual_zonal_mean(var, preds_3d, targets_3d, show=False, out_dir=save_path)

print('Done.')

# -*- coding: utf-8 -*-
import argparse
import h5py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os, sys, string
import cartopy.crs as ccrs

import matplotlib.ticker as ticker
import torch
from tqdm import tqdm
from climsim_utils.data_utils import *

from omegaconf import OmegaConf
from modulus import Module
from baseline_models.unet.training_default.joint_model import JointModel

parser = argparse.ArgumentParser()
parser.add_argument('--n_batches',          type=int,   default=None)
# --- optional diffusion model ---
parser.add_argument('--diff_config_path',   type=str,   default=None,
                    help='Path to saved_config.yaml for the joint diffusion model.')
parser.add_argument('--diff_checkpoint_path', type=str, default='',
                    help='Optional .mdlus checkpoint path (overrides diff_model.pt in config).')
parser.add_argument('--diff_n_batches',     type=int,   default=None,
                    help='Limit number of timestep batches for diffusion inference.')
parser.add_argument('--diff_start_doy',     type=int,   default=None,
                    help='Start day-of-year (0-indexed) for the diffusion npy data '
                         '(defaults to START_DOY used by the h5 data).')
args = parser.parse_args()

# --- Paths ---
grid_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc'

input_mean_v2_rh_mc_file = 'input_mean_v2_rh_mc_pervar.nc'
input_max_v2_rh_mc_file  = 'input_max_v2_rh_mc_pervar.nc'
input_min_v2_rh_mc_file  = 'input_min_v2_rh_mc_pervar.nc'
output_scale_v2_rh_mc_file = 'output_scale_std_lowerthred_v2_rh_mc.nc'

input_npy_path = "/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/val_set/val_input.npy"
target_npy_path = "/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/val_set/val_target.npy"

preds_path   = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/val_preds.h5'
targets_path = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/val_targets.h5'

save_path = '/global/homes/k/kfrields/climsim-kaggle-edition/figures/offline/residual_zonal_means'

# --- Grid and normalizations ---
print('Opening grid info...', flush=True)
grid_info = xr.open_dataset(grid_path)
grid_area = grid_info['area'].values
level     = grid_info.lev.values
lat       = grid_info['lat'].values
lon       = ((grid_info['lon'].values + 180) % 360) - 180

print('Opening input normalizations...', flush=True)
input_mean_v2_rh_mc  = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/'  + input_mean_v2_rh_mc_file)
input_max_v2_rh_mc   = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/'  + input_max_v2_rh_mc_file)
input_min_v2_rh_mc   = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/'  + input_min_v2_rh_mc_file)
print('Opening output normalizations...', flush=True)
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
print('Setting v2 rh mc vars...', flush=True)
data_utils_obj.set_to_v2_rh_mc_vars()

lat_bin_mids = data_utils_obj.lat_bin_mids
ncol         = grid_area.shape[0]

# --- Variable settings ---
# Each 3D output variable has 60 pressure levels; var_index is the first column in the flat output array.
vars_list = ['DTPHYS', 'DQ1PHYS', 'DQnPHYS', 'DUPHYS', 'DVPHYS']

def get_var_settings(var):
    if var == 'DTPHYS':
        var_title = 'Heating Tendency'
        unit      = 'K/s'
        var_index = 0
        vmin      = -5e-7
        vmax      =  5e-7
    elif var == 'DQ1PHYS':
        var_title = 'RH Tendency'
        unit      = ''
        var_index = 60
        vmin      = -1e-6
        vmax      =  1e-6
    elif var == 'DQnPHYS':
        var_title = 'Liquid+Ice Cloud Tendency'
        unit      = 'mg/kg/s'
        var_index = 120
        vmin      = -1e-3
        vmax      =  1e-3
    elif var == 'DUPHYS':
        var_title = 'Zonal Wind Tendency'
        unit      = 'm/s²'
        var_index = 180
        vmin      = -5e-7
        vmax      =  5e-7
    elif var == 'DVPHYS':
        var_title = 'Meridional Wind Tendency'
        unit      = 'm/s²'
        var_index = 240
        vmin      = -5e-7
        vmax      =  5e-7
    return var_title, unit, var_index

n_levels = 60

latitude_ticks  = [-60, -30, 0, 30, 60]
latitude_labels = ['60S', '30S', '0', '30N', '60N']

# --- Season helpers (20-min timesteps, no-leap 365-day calendar) ---
# ClimSim E3SM simulation starts at model year 0001-02-01 by default.
# Adjust START_DOY (0-indexed day-of-year) if your data begins on a different date.
TIMESTEP_MINUTES = 20
START_DOY        = 31   # Feb 1 → day index 31

_MONTH_DOY_START = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

def _timestep_month(t):
    """1-indexed calendar month for timestep index t (no-leap calendar)."""
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


def zonal_mean_da(arr_3d):
    """Area-weighted zonal mean of (time, ncol, lev) -> xr.DataArray (lev, lat)."""
    zm  = data_utils_obj.zonal_bin_weight_3d(arr_3d)  # (time, n_lat_bins, lev)
    zm  = zm.mean(axis=0)                              # (n_lat_bins, lev)
    return xr.DataArray(
        zm.T,
        dims=['hybrid pressure (hPa)', 'latitude'],
        coords={'hybrid pressure (hPa)': level, 'latitude': lat_bin_mids},
    )


def compute_zonal_stats(var, preds_ds, targets_ds, n_rows, n_time, time_mask=None):
    """Load only the 60 columns for `var` from the h5 datasets, then compute stats.

    Allocates (n_t, ncol, 60) per array instead of (n_t, ncol, n_output_vars),
    so memory scales with one variable at a time regardless of dataset width.
    """
    var_title, unit, var_index = get_var_settings(var)
    sl = slice(var_index, var_index + n_levels)

    pred_var   = preds_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)
    target_var = targets_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)

    if time_mask is not None:
        pred_var   = pred_var[time_mask]
        target_var = target_var[time_mask]

    residual = target_var - pred_var
    del pred_var, target_var
    bias_da = zonal_mean_da(residual)
    mae_da  = zonal_mean_da(np.abs(residual))
    del residual
    return bias_da, mae_da


def compute_diurnal_stats(var, preds_ds, targets_ds, n_rows, n_time):
    """Return (bias, mae) of shape (24, n_levels): area-weighted global mean per hour of day."""
    var_title, unit, var_index, vmin, vmax = get_var_settings(var)
    sl = slice(var_index, var_index + n_levels)

    pred_var   = preds_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)
    target_var = targets_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)
    residual   = target_var - pred_var
    del pred_var, target_var

    w = grid_area / grid_area.sum()                                          # (ncol,)
    res_gm  = (residual         * w[None, :, None]).sum(axis=1)              # (n_time, n_levels)
    ares_gm = (np.abs(residual) * w[None, :, None]).sum(axis=1)
    del residual

    minutes_abs = START_DOY * 1440 + np.arange(n_time) * TIMESTEP_MINUTES
    hour_of_day = (minutes_abs // 60) % 24                                   # integer 0-23

    bias_d = np.stack([res_gm [hour_of_day == h].mean(axis=0) for h in range(24)])  # (24, n_levels)
    mae_d  = np.stack([ares_gm[hour_of_day == h].mean(axis=0) for h in range(24)])
    del res_gm, ares_gm

    return bias_d, mae_d


def compute_column_mean_residual(var, preds_ds, targets_ds, n_rows, n_time, time_mask=None):
    """Vertical mean of residual (target - pred) → (n_t, ncol).
        n_rows is batch size * number of batches
    """
    var_title, unit, var_index, vmin, vmax = get_var_settings(var)
    sl = slice(var_index, var_index + n_levels)

    pred_var   = preds_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)
    target_var = targets_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)
    if time_mask is not None:
        pred_var   = pred_var[time_mask]
        target_var = target_var[time_mask]
    residual = (target_var - pred_var).mean(axis=2)
    del pred_var, target_var
    return residual


def compute_zonal_mean_direct(var, arr_flat, n_rows, n_time, time_mask=None):
    """Zonal mean of arr_flat directly (no target subtraction) → xr.DataArray (lev, lat)."""
    var_title, unit, var_index, vmin, vmax = get_var_settings(var)
    sl = slice(var_index, var_index + n_levels)
    data = arr_flat[:n_rows, sl].reshape(n_time, ncol, n_levels)
    if time_mask is not None:
        data = data[time_mask]
    return zonal_mean_da(data)


def compute_column_mean_direct(var, arr_flat, n_rows, n_time, time_mask=None):
    """Vertical mean of arr_flat directly (no target subtraction) → (n_t, ncol)."""
    var_title, unit, var_index, vmin, vmax = get_var_settings(var)
    sl = slice(var_index, var_index + n_levels)
    data = arr_flat[:n_rows, sl].reshape(n_time, ncol, n_levels)
    if time_mask is not None:
        data = data[time_mask]
    return data.mean(axis=2)


def plot_comparison_zonal_means(det_preds, targets, diff_preds, n_rows, n_time,
                                 time_mask=None,fname = 'comparison_zonal_means.png', title = 'True vs Predicted Residual',  show=True, out_dir=None):
    """Rows = variables, 3 cols = Robinson maps of (true residual | diff predicted | true MAE)."""
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
        var_title, unit, var_index, _, _ = get_var_settings(var)
        sl = slice(var_index, var_index + n_levels)

        pred_var = det_preds[:n_rows, sl].reshape(n_time, ncol, n_levels)
        targ_var = targets  [:n_rows, sl].reshape(n_time, ncol, n_levels)
        diff_var = diff_preds[:n_rows, sl].reshape(n_time, ncol, n_levels)
        if time_mask is not None:
            pred_var = pred_var[time_mask]
            targ_var = targ_var[time_mask]
            diff_var = diff_var[time_mask]

        residual = targ_var - pred_var
        true_map = residual.mean(axis=(0,2))
        diff_map = diff_var.mean(axis=(0,2))
        mae_map  = np.abs(residual).mean(axis=(0,2))
        del pred_var, targ_var, diff_var, residual

        #calculate top percentiles to use for scaling
        bias_max = float(np.nanpercentile(np.abs(true_map), 99))
        mae_max  = float(np.nanpercentile(mae_map, 99))

        cols = [
            (true_map, 'True Residual (Target − Det)', 'RdBu_r',  -bias_max, bias_max),
            (diff_map, 'Diff Predicted Residual',       'RdBu_r',  -bias_max, bias_max),
            (mae_map,  'True MAE (Det)',                 'viridis',  0,        mae_max),
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
            cbar = fig.colorbar(tc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label(unit, fontsize=9)
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


def plot_seasonal_bias_maps_comparison(det_preds, targets, diff_preds, n_rows, n_time,
                                        season_masks, lat, lon, out_dir=None):
    """One figure per season; rows = variables, 3 cols = (true residual | diff predicted | det MAE).

    Col 1 True residual  : column-mean of (target − det_pred)
    Col 2 Diff predicted : column-mean of diff_pred directly
    Col 3 True MAE       : column-mean of |target − det_pred|
    Cols 1 & 2 share a symmetric colorscale; col 3 uses viridis.
    """
    n_vars       = len(vars_list)
    panel_labels = [f'({l})' for l in string.ascii_lowercase[:n_vars * 3]]

    # Shared bias colorscale for cols 1 & 2 across all seasons; MAE scale separate
    print('  Computing shared color limits...', flush=True)
    var_bias_max = {}
    var_mae_max  = {}
    for var in vars_list:
        bias_vals, mae_vals = [], []
        var_title, unit, var_index, vmin, vmax = get_var_settings(var)
        sl = slice(var_index, var_index + n_levels)
        for mask in season_masks.values():
            if len(mask) == 0:
                continue
            true_res = compute_column_mean_residual(var, det_preds, targets,
                                                    n_rows, n_time, mask).mean(axis=0)
            diff_res = compute_column_mean_direct(var, diff_preds, n_rows, n_time, mask).mean(axis=0)
            pred_v   = det_preds[:n_rows, sl].reshape(n_time, ncol, n_levels)
            targ_v   = targets  [:n_rows, sl].reshape(n_time, ncol, n_levels)
            mae_res  = np.abs(targ_v[mask] - pred_v[mask]).mean(axis=2).mean(axis=0)
            bias_vals.append(true_res)
            mae_vals.append(mae_res)
            
        #calculate the 99th percentile of the absolute values of the biases and mae for color scaling
        var_bias_max[var] = float(np.nanpercentile(np.abs(np.concatenate(bias_vals)), 99))
        var_mae_max[var]  = float(np.nanpercentile(np.concatenate(mae_vals), 99))

    for key, info in SEASONS.items():
        mask = season_masks[key]
        if len(mask) == 0:
            continue

        fig, axs = plt.subplots(
            n_vars, 3,
            figsize=(24, 3.5 * n_vars),
            subplot_kw={'projection': ccrs.Robinson()},
            constrained_layout=True,
        )
        if n_vars == 1:
            axs = axs[np.newaxis, :]

        for row, var in enumerate(vars_list):
            var_title, unit, var_index, vmin, vmax = get_var_settings(var)
            sl       = slice(var_index, var_index + n_levels)
            bias_max = var_bias_max[var]
            mae_max  = var_mae_max[var]

            true_res = compute_column_mean_residual(var, det_preds, targets,
                                                    n_rows, n_time, mask).mean(axis=0)
            diff_res = compute_column_mean_direct(var, diff_preds, n_rows, n_time, mask).mean(axis=0)
            pred_v   = det_preds[:n_rows, sl].reshape(n_time, ncol, n_levels)
            targ_v   = targets  [:n_rows, sl].reshape(n_time, ncol, n_levels)
            mae_res  = np.abs(targ_v[mask] - pred_v[mask]).mean(axis=2).mean(axis=0)

            cols = [
                (true_res, 'True Residual (Target − Det)', 'RdBu_r',  -bias_max, bias_max),
                (diff_res, 'Diff Predicted Residual',       'RdBu_r',  -bias_max, bias_max),
                (mae_res,  'True MAE (Det)',                 'viridis',  0,        mae_max),
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
                cbar = fig.colorbar(tc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                cbar.set_label(unit, fontsize=9)
                cbar.locator = ticker.MaxNLocator(nbins=4)

        fig.suptitle(f'True vs Predicted Residual — {info["label"]}', fontsize=11)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fpath = os.path.join(out_dir, f'comparison_bias_map_{key.lower()}.png')
            plt.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f'Saved: {fpath}', flush=True)
        plt.close()


def plot_seasonal_bias_maps(preds_ds, targets_ds, n_rows, n_time, season_masks, lat, lon,
                             out_dir=None):
    """One figure per season; rows = variables, map = tiled column-mean bias."""
    n_vars       = len(vars_list)
    panel_labels = [f'({l})' for l in string.ascii_lowercase[:n_vars]]

    # Compute shared color limits per variable across all seasons
    print('  Computing shared color limits across seasons...', flush=True)
    var_abs_max = {}
    for var in vars_list:
        all_biases = []
        for mask in season_masks.values():
            if len(mask) == 0:
                continue
            col_bias = compute_column_mean_residual(var, preds_ds, targets_ds,
                                                    n_rows, n_time, time_mask=mask)
            all_biases.append(col_bias.mean(axis=0))
        var_abs_max[var] = float(np.nanpercentile(np.abs(np.concatenate(all_biases)), 99))

    for key, info in SEASONS.items():
        mask = season_masks[key]
        if len(mask) == 0:
            print(f'  No timesteps for {key}, skipping maps.', flush=True)
            continue

        fig, axs = plt.subplots(
            n_vars, 1,
            figsize=(10, 3.5 * n_vars),
            subplot_kw={'projection': ccrs.Robinson()},
            constrained_layout=True,
        )
        if n_vars == 1:
            axs = [axs]

        for row, var in enumerate(vars_list):
            var_title, unit, var_index, vmin, vmax = get_var_settings(var)
            col_bias  = compute_column_mean_residual(var, preds_ds, targets_ds,
                                                     n_rows, n_time, time_mask=mask)
            mean_bias = col_bias.mean(axis=0)  # (ncol,)
            abs_max   = var_abs_max[var]

            ax = axs[row]
            levels_c = np.linspace(-abs_max, abs_max, 20)
            tc = ax.tricontourf(lon, lat, mean_bias, transform=ccrs.PlateCarree(),
                                cmap='RdBu_r', levels=levels_c, extend='both',
                                vmin=-abs_max, vmax=abs_max)
            ax.coastlines(linewidth=0.5, color='black')
            ax.set_global()
            ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            ax.set_title(f"{panel_labels[row]} {var_title} — Mean Bias (Target − Pred)", fontsize=9)
            cbar = fig.colorbar(tc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label(unit, fontsize=9)
            cbar.locator = ticker.MaxNLocator(nbins=4)

        fig.suptitle(f'Seasonal Mean Column Bias — {info["label"]}', fontsize=11)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fpath = os.path.join(out_dir, f'seasonal_bias_map_{key.lower()}.png')
            plt.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f'Saved: {fpath}', flush=True)
        plt.close()


def plot_diurnal_cycle(preds_ds, targets_ds, n_rows, n_time,
                        title='Diurnal Cycle of Residuals',
                        fname='residual_diurnal_cycle.png', show=True, out_dir=None):
    """Rows=variables, cols=(bias, MAE). x=hour of day (UTC), y=pressure level."""
    n_vars = len(vars_list)
    hours  = np.arange(24)
    labels = [f'({l})' for l in string.ascii_lowercase[:n_vars * 2]]

    fig, axs = plt.subplots(n_vars, 2, figsize=(9, 2.8 * n_vars), constrained_layout=True)

    for row, var in enumerate(vars_list):
        var_title, unit, *_ = get_var_settings(var)
        bias_d, mae_d = compute_diurnal_stats(var, preds_ds, targets_ds, n_rows, n_time)

        bias_abs_max = float(np.nanmax(np.abs(bias_d)))

        bias_da = xr.DataArray(bias_d.T, dims=['hybrid pressure (hPa)', 'hour'],
                               coords={'hybrid pressure (hPa)': level, 'hour': hours})
        mae_da  = xr.DataArray(mae_d.T,  dims=['hybrid pressure (hPa)', 'hour'],
                               coords={'hybrid pressure (hPa)': level, 'hour': hours})

        ax0 = axs[row, 0]
        im0 = bias_da.plot(ax=ax0, add_colorbar=False, cmap='RdBu_r',
                           vmin=-bias_abs_max, vmax=bias_abs_max)
        fig.colorbar(im0, ax=ax0, label=unit, pad=0.02)
        ax0.set_title(f"{labels[row*2]} {var_title} — Mean Bias (Target − Pred)", fontsize=8)
        ax0.invert_yaxis()
        ax0.set_xlabel('Hour of Day (UTC)', fontsize=7)
        ax0.set_ylabel('Hybrid pressure (hPa)', fontsize=7)
        ax0.set_xticks(np.arange(0, 24, 6))
        ax0.tick_params(labelsize=7)

        ax1 = axs[row, 1]
        im1 = mae_da.plot(ax=ax1, add_colorbar=False, cmap='viridis')
        fig.colorbar(im1, ax=ax1, label=unit, pad=0.02)
        ax1.set_title(f"{labels[row*2+1]} {var_title} — MAE", fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel('Hour of Day (UTC)', fontsize=7)
        ax1.set_ylabel('', fontsize=7)
        ax1.set_xticks(np.arange(0, 24, 6))
        ax1.tick_params(labelsize=7)

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


def plot_all_residual_zonal_means(preds_ds, targets_ds, n_rows, n_time, time_mask=None,
                                   title='Zonal Mean Residuals',
                                   fname='residual_zonal_means_all.png', show=True, out_dir=None):
    """Single figure with all variables; rows=variables, cols=(bias, MAE)."""
    n_vars = len(vars_list)
    labels = [f'({l})' for l in string.ascii_lowercase[:n_vars * 2]]

    fig, axs = plt.subplots(n_vars, 2, figsize=(9, 2.8 * n_vars),
                            constrained_layout=True)

    for row, var in enumerate(vars_list):
        var_title, unit, *_ = get_var_settings(var)
        bias_da, mae_da = compute_zonal_stats(var, preds_ds, targets_ds, n_rows, n_time,
                                              time_mask=time_mask)

        bias_abs_max = float(np.nanmax(np.abs(bias_da.values)))

        # Bias
        ax0 = axs[row, 0]
        im0 = bias_da.plot(ax=ax0, add_colorbar=False, cmap='RdBu_r',
                           vmin=-bias_abs_max, vmax=bias_abs_max)
        fig.colorbar(im0, ax=ax0, label=unit, pad=0.02)
        ax0.set_title(f"{labels[row*2]} {var_title} — Mean Bias (Target − Pred)", fontsize=8)
        ax0.invert_yaxis()
        ax0.set_xlabel('Latitude', fontsize=7)
        ax0.set_ylabel('Hybrid pressure (hPa)', fontsize=7)
        ax0.set_xticks(latitude_ticks)
        ax0.set_xticklabels(latitude_labels, fontsize=7)
        ax0.tick_params(axis='y', labelsize=7)

        # MAE
        ax1 = axs[row, 1]
        im1 = mae_da.plot(ax=ax1, add_colorbar=False, cmap='viridis')
        fig.colorbar(im1, ax=ax1, label=unit, pad=0.02)
        ax1.set_title(f"{labels[row*2+1]} {var_title} — MAE", fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel('Latitude', fontsize=7)
        ax1.set_ylabel('', fontsize=7)
        ax1.set_xticks(latitude_ticks)
        ax1.set_xticklabels(latitude_labels, fontsize=7)
        ax1.tick_params(axis='y', labelsize=7)

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


# ---------------------------------------------------------------------------
# Diffusion model loading and inference
# ---------------------------------------------------------------------------

def load_joint_model(config_path, checkpoint_path):
    """Load joint (deterministic + diffusion) model and preprocess npy data.

    Returns
    -------
    joint_model   : JointModel on CUDA/CPU
    torch_input   : FloatTensor  (n_total, input_features)  — normalised inputs
    targets_flat  : ndarray      (n_time*ncol, n_output)    — raw physical targets
    n_time_diff   : int
    diff_data     : data_utils instance
    out_scale_np  : ndarray  (1, n_output)  — output std for denormalisation
    """
    # Project root needs to be on sys.path for physicsnemo + baseline_models
    #project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    #if project_root not in sys.path:
    #    sys.path.insert(0, project_root)

    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config_path) as f:
        cfg = OmegaConf.load(f)

    # --- normalisations ---
    gi   = xr.open_dataset(cfg.grid_info_path)
    imn  = xr.open_dataset(cfg.input_mean_path)
    imx  = xr.open_dataset(cfg.input_max_path)
    imi  = xr.open_dataset(cfg.input_min_path)
    osc  = xr.open_dataset(cfg.output_scale_path)
    qn_lbd = np.loadtxt(cfg.qn_lbd_path, delimiter=',')

    res_std    = torch.load(cfg.res_std_path,   map_location=device).to(torch.float32)
    res_mean   = torch.load(cfg.res_mean_path,  map_location=device).to(torch.float32)
    preds_std  = torch.load(cfg.preds_std_path, map_location=device).to(torch.float32)
    preds_mean = torch.load(cfg.preds_mean_path,map_location=device).to(torch.float32)

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
    out_scale_np = out_scale[None, :]          # (1, n_output)

    # --- preprocess npy inputs (mirrors preprocessing_v2_rh_mc in notebook) ---
    print(f'  np.load input  ({input_npy_path})...', flush=True)
    npy_input  = np.load(input_npy_path)
    print(f'  np.load target ({target_npy_path})...', flush=True)
    npy_target = np.load(target_npy_path)
    print(f'  Preprocessing input array (shape {npy_input.shape})...', flush=True)

    npy_input[:, 120:180] = 1 - np.exp(-npy_input[:, 120:180] * qn_lbd)
    npy_input = (npy_input - input_sub) / input_div
    npy_input = np.where(np.isnan(npy_input), 0, npy_input)
    npy_input = np.where(np.isinf(npy_input), 0, npy_input)
    npy_input[:, 120:135]  = 0
    npy_input[:, 60:120]   = np.clip(npy_input[:, 60:120], 0, 1.2)
    print('  Converting to torch tensor...', flush=True)
    torch_input = torch.tensor(npy_input).float()

    ncol_diff    = diff_data.num_latlon
    n_time_diff  = npy_input.shape[0] // ncol_diff
    targets_flat = npy_target[:n_time_diff * ncol_diff]   # (n_time*ncol, n_output)

    # --- deterministic model ---
    base = os.path.join(cfg.save_path, cfg.expname)
    print(f'  Loading deterministic model ({os.path.join(base, "unet_model.pt")})...', flush=True)
    det_model = torch.jit.load(os.path.join(base, 'unet_model.pt')).to(device)
    det_model.eval()

    # --- diffusion model ---
    if checkpoint_path:
        print(f'  Loading diffusion model from checkpoint ({checkpoint_path})...', flush=True)
        diff_model = Module.from_checkpoint(checkpoint_path).to(device)
    else:
        print(f'  Loading diffusion model ({os.path.join(base, "diff_model.pt")})...', flush=True)
        diff_model = torch.load(os.path.join(base, 'diff_model.pt')).to(device)
    diff_model.eval()
    print('  Models loaded.', flush=True)

    # --- condition channel count (mirrors notebook logic exactly) ---
    loc   = cfg.diffusion_model.condition_location
    ctype = cfg.diffusion_model.condition_type
    base_ch = (diff_data.target_profile_num + diff_data.target_scalar_num +
               diff_data.input_profile_num  + diff_data.input_scalar_num)
    if loc == 'front':
        cond_channels = base_ch if ctype == 'input_output' else base_ch
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
    """Run joint model inference over all timesteps.

    Returns
    -------
    joint_preds_flat : ndarray  (n_time*ncol, n_output)  — det + diffusion
    det_preds_flat   : ndarray  (n_time*ncol, n_output)  — deterministic only
    """
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

            # deterministic forward
            output, _ = joint_model.deterministic_model(input_batch)

            # build conditioning (mirrors inference_joint_model in notebook)
            safe_std       = torch.clamp(joint_model.res_std, min=1e-2)
            cond_out       = ((output - joint_model.preds_mean) /
                              (joint_model.preds_std + 1e-8)) * 0.5
            loc = joint_model.condition_location
            if loc == 'front' and joint_model.condition_type == 'input_output':
                condition_data = torch.cat((input_batch, cond_out), dim=1)
            if loc == 'embedding' and joint_model.condition_type == 'input_output':
                lc = torch.cat((input_batch, cond_out), dim=1)
                condition_data = lc.reshape(lc.shape[0], -1)
            elif loc in ('middle', 'cross') and joint_model.condition_type == 'input_output':
                condition_data = torch.cat((input_batch, cond_out), dim=1)

            # diffusion sampling
            latents = torch.randn(
                (batch_size, diff_data.target_profile_num + diff_data.target_scalar_num, 64),
                device=device)
            res = joint_model.res_model.edm_sampler(
                latents, condition_input=condition_data,
                sigma_min=sigma_min, sigma_max=sigma_max,
                rho=rho, num_steps=num_steps)

            denorm_res  = (res / 0.5) * (safe_std + 1e-8)
            reshaped_res = joint_model.reverse_reshape_target(denorm_res)
            reshaped_out = joint_model.reverse_reshape_target(output)

            joint_pred = reshaped_out + reshaped_res
            joint_pred[:, 300:] = torch.nn.functional.relu(joint_pred[:, 300:])

            det_list.append(  (reshaped_out.cpu().numpy()  / out_scale_np))
            joint_list.append((joint_pred.cpu().numpy()    / out_scale_np))
            diff_list.append( (reshaped_res.cpu().numpy()  / out_scale_np))
            n_done += 1

    det_preds_flat   = np.concatenate(det_list,   axis=0)   # (n*ncol, 308)
    joint_preds_flat = np.concatenate(joint_list, axis=0)
    diff_preds_flat  = np.concatenate(diff_list,  axis=0)   # raw diffusion residual
    return joint_preds_flat, det_preds_flat, diff_preds_flat


# ---------------------------------------------------------------------------
# Optional: diffusion model inference + comparison visualisations (runs first)
# ---------------------------------------------------------------------------
if args.diff_config_path:

    print('\n=== Diffusion model ===', flush=True)
    print('Loading joint model...', flush=True)
    joint_model, torch_input_diff, targets_flat_diff, n_time_diff, diff_data, out_scale_diff = \
        load_joint_model(
            config_path      = args.diff_config_path,
            checkpoint_path  = args.diff_checkpoint_path,
        )

    print('Running inference...', flush=True)
    joint_preds_flat, det_preds_flat, diff_preds_flat = run_diffusion_inference(
        joint_model, diff_data, torch_input_diff, out_scale_diff,
        n_batches_limit = args.diff_n_batches,
    )

    # n_time may be truncated if --diff_n_batches was set
    ncol_diff   = diff_data.num_latlon
    n_time_used = joint_preds_flat.shape[0] // ncol_diff
    n_rows_diff = n_time_used * ncol_diff

    # Build seasonal masks using the same calendar logic
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
    os.makedirs(diff_save_path, exist_ok=True)

    # --- Annual comparison: true vs predicted residual ---
    print('Plotting annual comparison zonal means...', flush=True)
    plot_comparison_zonal_means(
        det_preds_flat, targets_flat_diff, diff_preds_flat, n_rows_diff, n_time_used,
        title   = 'True vs Predicted Residual — Annual',
        fname   = 'comparison_zonal_means_annual.png',
        show    = False, out_dir = diff_save_path,
    )

    # --- Seasonal comparisons ---
    print('Plotting seasonal comparison zonal means...', flush=True)
    for key, info in SEASONS.items():
        mask = diff_season_masks[key]
        if len(mask) == 0:
            print(f'  No timesteps for {key}, skipping.', flush=True)
            continue
        print(f'  {info["label"]} ({len(mask)} timesteps)...', flush=True)
        plot_comparison_zonal_means(
            det_preds_flat, targets_flat_diff, diff_preds_flat, n_rows_diff, n_time_used,
            time_mask = mask,
            title     = f'True vs Predicted Residual — {info["label"]}',
            fname     = f'comparison_zonal_means_{key.lower()}.png',
            show      = False, out_dir = diff_save_path,
        )

    print('Plotting seasonal comparison bias maps...', flush=True)
    plot_seasonal_bias_maps_comparison(
        det_preds_flat, targets_flat_diff, diff_preds_flat, n_rows_diff, n_time_used,
        diff_season_masks, lat, lon,
        out_dir = diff_save_path,
    )

    print('=== Diffusion model done ===\n', flush=True)

# --- Open h5 files and keep them open for lazy per-variable loading ---
n_batches_to_load = args.n_batches

print('Opening h5 files...', flush=True)
with h5py.File(preds_path, 'r') as preds_f, h5py.File(targets_path, 'r') as targets_f:
    preds_ds   = preds_f['data'] #shape (time*batches, features)
    targets_ds = targets_f['data']

    n_output_vars = preds_ds.shape[1]
    n_time_total  = preds_ds.shape[0] // ncol
    n_time        = min(n_batches_to_load, n_time_total) if n_batches_to_load else n_time_total
    n_rows        = n_time * ncol
    print(f'ncol={ncol}, n_time={n_time}/{n_time_total}, n_output_vars={n_output_vars}', flush=True)

    # --- Build seasonal masks (pure index math, no data loaded yet) ---
    months = np.array([_timestep_month(t) for t in range(n_time)])
    season_masks = {
        key: np.where(np.isin(months, list(info['months'])))[0]
        for key, info in SEASONS.items()
    }

    # --- Plot annual ---
    print('Plotting annual...', flush=True)
    plot_all_residual_zonal_means(preds_ds, targets_ds, n_rows, n_time,
                                   show=False, out_dir=save_path)

    # --- Plot per season ---
    for key, info in SEASONS.items():
        mask = season_masks[key]
        if len(mask) == 0:
            print(f'  No timesteps for {key}, skipping.', flush=True)
            continue
        print(f'Plotting {info["label"]} ({len(mask)} timesteps)...', flush=True)
        plot_all_residual_zonal_means(
            preds_ds, targets_ds, n_rows, n_time,
            time_mask=mask,
            title=f'Zonal Mean Residuals — {info["label"]}',
            fname=f'residual_zonal_means_{key.lower()}.png',
            show=False, out_dir=save_path,
        )

    # --- Plot diurnal cycle ---
    print('Plotting diurnal cycle...', flush=True)
    plot_diurnal_cycle(preds_ds, targets_ds, n_rows, n_time,
                        show=False, out_dir=save_path)

    # --- Plot seasonal bias maps ---
    print('Plotting seasonal bias maps...', flush=True)
    plot_seasonal_bias_maps(preds_ds, targets_ds, n_rows, n_time, season_masks, lat, lon,
                             out_dir=save_path)

print('All done.', flush=True)

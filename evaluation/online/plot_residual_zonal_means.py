# -*- coding: utf-8 -*-
import argparse
import h5py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os, string
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as mtri
from climsim_utils.data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_batches', type=int, default=None)
args = parser.parse_args()

# --- Paths ---
grid_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc'

input_mean_v2_rh_mc_file = 'input_mean_v2_rh_mc_pervar.nc'
input_max_v2_rh_mc_file  = 'input_max_v2_rh_mc_pervar.nc'
input_min_v2_rh_mc_file  = 'input_min_v2_rh_mc_pervar.nc'
output_scale_v2_rh_mc_file = 'output_scale_std_lowerthred_v2_rh_mc.nc'

preds_path   = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/train_preds.h5'
targets_path = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/train_targets.h5'

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
var_settings = {
    'DTPHYS':  {'var_title': 'Heating Tendency',          'scaling': 1.,   'unit': 'K/s',      'var_index': 0,   'vmax': 5e-7,  'vmin': -5e-7},
    'DQ1PHYS': {'var_title': 'RH Tendency',               'scaling': 1.,   'unit': '',         'var_index': 60,  'vmax': 1e-6,  'vmin': -1e-6},
    'DQnPHYS': {'var_title': 'Liquid+Ice Cloud Tendency', 'scaling': 1e6,  'unit': 'mg/kg/s',  'var_index': 120, 'vmax': 1e-3,  'vmin': -1e-3},
    'DUPHYS':  {'var_title': 'Zonal Wind Tendency',       'scaling': 1.,   'unit': 'm/s²',     'var_index': 180, 'vmax': 5e-7,  'vmin': -5e-7},
    'DVPHYS':  {'var_title': 'Meridional Wind Tendency',  'scaling': 1.,   'unit': 'm/s²',     'var_index': 240, 'vmax': 5e-7,  'vmin': -5e-7},
}

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
    s   = var_settings[var]
    sc  = s['scaling']
    idx = s['var_index']
    sl  = slice(idx, idx + n_levels)

    pred_var   = preds_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)
    target_var = targets_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)

    if time_mask is not None:
        pred_var   = pred_var[time_mask]
        target_var = target_var[time_mask]

    residual = target_var - pred_var
    del pred_var, target_var
    bias_da = sc * zonal_mean_da(residual)
    mae_da  = sc * zonal_mean_da(np.abs(residual))
    del residual
    return bias_da, mae_da


def compute_diurnal_stats(var, preds_ds, targets_ds, n_rows, n_time):
    """Return (bias, mae) of shape (24, n_levels): area-weighted global mean per hour of day."""
    s   = var_settings[var]
    sc  = s['scaling']
    idx = s['var_index']
    sl  = slice(idx, idx + n_levels)

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

    return sc * bias_d, sc * mae_d


def compute_column_mean_residual(var, preds_ds, targets_ds, n_rows, n_time, time_mask=None):
    """Vertical mean of residual (target - pred) → (n_t, ncol)."""
    s   = var_settings[var]
    idx = s['var_index']
    sl  = slice(idx, idx + n_levels)

    pred_var   = preds_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)
    target_var = targets_ds[:n_rows, sl].reshape(n_time, ncol, n_levels)
    if time_mask is not None:
        pred_var   = pred_var[time_mask]
        target_var = target_var[time_mask]
    residual = (target_var - pred_var).mean(axis=2)
    del pred_var, target_var
    return s['scaling'] * residual


def _make_triangulation(lon, lat):
    """Delaunay triangulation with antimeridian-spanning triangles masked out."""
    tri = mtri.Triangulation(lon, lat)
    # mask triangles whose vertices straddle the antimeridian (lon gap > 180°)
    lons = lon[tri.triangles]
    mask = (lons.max(axis=1) - lons.min(axis=1)) > 180
    tri.set_mask(mask)
    return tri


def plot_seasonal_bias_maps(preds_ds, targets_ds, n_rows, n_time, season_masks, lat, lon,
                             out_dir=None):
    """One figure per season; rows = variables, map = tiled column-mean bias."""
    vars_list    = list(var_settings.keys())
    n_vars       = len(vars_list)
    panel_labels = [f'({l})' for l in string.ascii_lowercase[:n_vars]]

    tri = _make_triangulation(lon, lat)

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
            s         = var_settings[var]
            col_bias  = compute_column_mean_residual(var, preds_ds, targets_ds,
                                                     n_rows, n_time, time_mask=mask)
            mean_bias = col_bias.mean(axis=0)  # (ncol,)
            abs_max   = var_abs_max[var]

            ax = axs[row]
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.set_global()
            tc = ax.tripcolor(tri, mean_bias,
                              cmap='RdBu_r', vmin=-abs_max, vmax=abs_max,
                              transform=ccrs.PlateCarree(), rasterized=True)
            fig.colorbar(tc, ax=ax, orientation='horizontal', pad=0.04,
                         shrink=0.7, label=s['unit'])
            ax.set_title(f"{panel_labels[row]} {s['var_title']} — Mean Bias (Target − Pred)",
                         fontsize=9)

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
    vars_list = list(var_settings.keys())
    n_vars    = len(vars_list)
    hours     = np.arange(24)
    labels    = [f'({l})' for l in string.ascii_lowercase[:n_vars * 2]]

    fig, axs = plt.subplots(n_vars, 2, figsize=(9, 2.8 * n_vars), constrained_layout=True)

    for row, var in enumerate(vars_list):
        s = var_settings[var]
        bias_d, mae_d = compute_diurnal_stats(var, preds_ds, targets_ds, n_rows, n_time)

        bias_abs_max = float(np.nanmax(np.abs(bias_d)))

        bias_da = xr.DataArray(bias_d.T, dims=['hybrid pressure (hPa)', 'hour'],
                               coords={'hybrid pressure (hPa)': level, 'hour': hours})
        mae_da  = xr.DataArray(mae_d.T,  dims=['hybrid pressure (hPa)', 'hour'],
                               coords={'hybrid pressure (hPa)': level, 'hour': hours})

        ax0 = axs[row, 0]
        im0 = bias_da.plot(ax=ax0, add_colorbar=False, cmap='RdBu_r',
                           vmin=-bias_abs_max, vmax=bias_abs_max)
        fig.colorbar(im0, ax=ax0, label=s['unit'], pad=0.02)
        ax0.set_title(f"{labels[row*2]} {s['var_title']} — Mean Bias (Target − Pred)", fontsize=8)
        ax0.invert_yaxis()
        ax0.set_xlabel('Hour of Day (UTC)', fontsize=7)
        ax0.set_ylabel('Hybrid pressure (hPa)', fontsize=7)
        ax0.set_xticks(np.arange(0, 24, 6))
        ax0.tick_params(labelsize=7)

        ax1 = axs[row, 1]
        im1 = mae_da.plot(ax=ax1, add_colorbar=False, cmap='viridis')
        fig.colorbar(im1, ax=ax1, label=s['unit'], pad=0.02)
        ax1.set_title(f"{labels[row*2+1]} {s['var_title']} — MAE", fontsize=8)
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
    vars_list = list(var_settings.keys())
    n_vars    = len(vars_list)
    labels    = [f'({l})' for l in string.ascii_lowercase[:n_vars * 2]]

    fig, axs = plt.subplots(n_vars, 2, figsize=(9, 2.8 * n_vars),
                            constrained_layout=True)

    for row, var in enumerate(vars_list):
        s = var_settings[var]
        bias_da, mae_da = compute_zonal_stats(var, preds_ds, targets_ds, n_rows, n_time,
                                              time_mask=time_mask)

        bias_abs_max = float(np.nanmax(np.abs(bias_da.values)))

        # Bias
        ax0 = axs[row, 0]
        im0 = bias_da.plot(ax=ax0, add_colorbar=False, cmap='RdBu_r',
                           vmin=-bias_abs_max, vmax=bias_abs_max)
        fig.colorbar(im0, ax=ax0, label=f"{s['unit']}", pad=0.02)
        ax0.set_title(f"{labels[row*2]} {s['var_title']} — Mean Bias (Target − Pred)", fontsize=8)
        ax0.invert_yaxis()
        ax0.set_xlabel('Latitude', fontsize=7)
        ax0.set_ylabel('Hybrid pressure (hPa)', fontsize=7)
        ax0.set_xticks(latitude_ticks)
        ax0.set_xticklabels(latitude_labels, fontsize=7)
        ax0.tick_params(axis='y', labelsize=7)

        # MAE
        ax1 = axs[row, 1]
        im1 = mae_da.plot(ax=ax1, add_colorbar=False, cmap='viridis')
        fig.colorbar(im1, ax=ax1, label=f"{s['unit']}", pad=0.02)
        ax1.set_title(f"{labels[row*2+1]} {s['var_title']} — MAE", fontsize=8)
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


# --- Open h5 files and keep them open for lazy per-variable loading ---
n_batches_to_load = args.n_batches

print('Opening h5 files...', flush=True)
with h5py.File(preds_path, 'r') as preds_f, h5py.File(targets_path, 'r') as targets_f:
    preds_ds   = preds_f['data']
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

print('Done.', flush=True)

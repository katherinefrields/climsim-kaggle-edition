# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
#import pandas as pd
#from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#import matplotlib.ticker as ticker
#from matplotlib.cm import ScalarMappable
#from matplotlib import gridspec
import os,  glob, string #argparse, gc, sys
from tqdm import tqdm
#import time
#import itertools
#import sys
#import pickle
#import cartopy
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
from climsim_utils.data_utils import *

grid_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc'

input_mean_v2_rh_mc_file = 'input_mean_v2_rh_mc_pervar.nc'

input_max_v2_rh_mc_file = 'input_max_v2_rh_mc_pervar.nc'
input_min_v2_rh_mc_file = 'input_min_v2_rh_mc_pervar.nc'
output_scale_v2_rh_mc_file = 'output_scale_std_lowerthred_v2_rh_mc.nc'

lbd_qn_file = 'qn_exp_lambda_large.txt'

grid_info = xr.open_dataset(grid_path)
grid_area = grid_info['area'].values
area_weight = grid_area/np.sum(grid_area)
level = grid_info.lev.values

input_mean_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_mean_v2_rh_mc_file)
input_max_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_max_v2_rh_mc_file)
input_min_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + input_min_v2_rh_mc_file)
output_scale_v2_rh_mc = xr.open_dataset('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/outputs/' + output_scale_v2_rh_mc_file)

lbd_qn = np.loadtxt('/global/cfs/cdirs/m4334/jerry/climsim3_dev/preprocessing/normalizations/inputs/' + lbd_qn_file, delimiter = ',')

data_v2_rh_mc = data_utils(grid_info = grid_info, 
                           input_mean = input_mean_v2_rh_mc, 
                           input_max = input_max_v2_rh_mc, 
                           input_min = input_min_v2_rh_mc, 
                           output_scale = output_scale_v2_rh_mc,
                           qinput_log = False,
                           normalize = False)
data_v2_rh_mc.set_to_v2_rh_mc_vars()

actual_input_v2_rh_mc = np.load('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/scoring_set/scoring_input.npy')
actual_target_v2_rh_mc = np.load('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/scoring_set/scoring_target.npy')
ncol = grid_area.shape[0]
actual_input_v2_rh_mc = actual_input_v2_rh_mc.reshape(-1, ncol, actual_input_v2_rh_mc.shape[-1])
actual_target_v2_rh_mc = actual_target_v2_rh_mc.reshape(-1, ncol, actual_target_v2_rh_mc.shape[-1])

actual_target = np.load('/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/scoring_set/scoring_target.npy')
del actual_target_v2_rh_mc

surface_pressure = actual_input_v2_rh_mc[:, :, data_v2_rh_mc.ps_index]
hyam_component = (data_v2_rh_mc.hyam * data_v2_rh_mc.p0)[None,None,:]
hybm_component = data_v2_rh_mc.hybm[None,None,:] * surface_pressure[:,:,None]
pressures = hyam_component + hybm_component
pressures_binned = data_v2_rh_mc.zonal_bin_weight_3d(pressures)
lat_bin_mids = data_v2_rh_mc.lat_bin_mids
lats = data_v2_rh_mc.lats
lons = data_v2_rh_mc.lons

idx_p400_t10 = np.load('/pscratch/sd/z/zeyuanhu/hu_etal2024_data/microphysics_hourly/first_true_indices_p400_t10.npy')
for i in range(idx_p400_t10.shape[0]):
    for j in range(idx_p400_t10.shape[1]):
        idx_p400_t10[i,j] = level[int(idx_p400_t10[i,j])]

idx_p400_t10 = idx_p400_t10.mean(axis=0)
idx_p400_t10 = idx_p400_t10[np.newaxis,:]

idx_tropopause_zm = data_v2_rh_mc.zonal_bin_weight_2d(idx_p400_t10).flatten()

area_weight_dict = {
    'global': area_weight,
    'nh': np.where(lats > 30, area_weight, 0),
    'sh': np.where(lats < -30, area_weight, 0),
    'tropics': np.where((lats > -30) & (lats < 30), area_weight, 0)
}

lat_idx_dict = {
    '30S_30N': ((data_v2_rh_mc.lats < 30) & (data_v2_rh_mc.lats > -30))[None,:,None],
    '30N_60N': ((data_v2_rh_mc.lats < 60) & (data_v2_rh_mc.lats > 30))[None,:,None],
    '30S_60S': ((data_v2_rh_mc.lats < -30) & (data_v2_rh_mc.lats > -60))[None,:,None],
    '60N_90N': (data_v2_rh_mc.lats > 60)[None,:,None],
    '60S_90S': (data_v2_rh_mc.lats < -60)[None,:,None]
}

pressure_idx_dict = {
    'below_400hPa': pressures >= 400,
    'above_400hPa': pressures < 400
}

config_names = {
    'joint': 'Joint',
    'unet': 'Deterministic U-Net'
}

model_names = {
    'joint': 'Joint',
    'unet': 'U-Net',
}

color_dict = {
    'joint': 'blue',
    'unet': 'green',
}

color_dict_config = {
    'joint': 'blue',
    'unet': 'cyan',
}

offline_var_settings = {
    'DTPHYS': {'var_title': 'dT/dt', 'scaling': 1., 'unit': 'K/s', 'vmax': 5e-7, 'vmin': -5e-7, 'var_index':0},
    'DQ1PHYS': {'var_title': 'dQv/dt', 'scaling': 1e3, 'unit': 'g/kg/s', 'vmax': 1e-6, 'vmin': -1e-6, 'var_index':60},
    'DQ2PHYS': {'var_title': 'dQl/dt', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 1e-3, 'vmin': -1e-3, 'var_index':120},
    'DQ3PHYS': {'var_title': 'dQi/dt', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 1e-3, 'vmin': -1e-3, 'var_index':180},
    'DUPHYS': {'var_title': 'dU/dt', 'scaling': 1., 'unit': 'm/s/s', 'vmax': 5e-7, 'vmin': -5e-7, 'var_index':240},
    'DVPHYS': {'var_title': 'dV/dt', 'scaling': 1., 'unit': 'm/s/s', 'vmax': 5e-7, 'vmin': -5e-7, 'var_index':300}
}

online_var_settings = {
    'T': {'var_title': 'Temperature', 'scaling': 1.0, 'unit': 'K', 'vmax': 5, 'vmin': -5},
    'Q': {'var_title': 'Specific Humidity', 'scaling': 1000.0, 'unit': 'g/kg', 'vmax': 1, 'vmin': -1},
    'U': {'var_title': 'Zonal Wind', 'scaling': 1.0, 'unit': 'm/s', 'vmax': 4, 'vmin': -4},
    'V': {'var_title': 'Meridional Wind', 'scaling': 1.0, 'unit': 'm/s', 'vmax': 4, 'vmin': -4},
    'CLDLIQ': {'var_title': 'Liquid Cloud', 'scaling': 1e6, 'unit': 'mg/kg', 'vmax': 40, 'vmin': -40},
    'CLDICE': {'var_title': 'Ice Cloud', 'scaling': 1e6, 'unit': 'mg/kg', 'vmax': 5, 'vmin': -5},
    'TOTCLD': {'var_title': 'Total Cloud', 'scaling': 1e6, 'unit': 'mg/kg', 'vmax': 40, 'vmin': -40},
    'DTPHYS': {'var_title': 'Heating Tendency', 'scaling': 1., 'unit': 'K/s', 'vmax': 1.5e-5, 'vmin': -1.5e-5},
    'DQ1PHYS': {'var_title': 'Moistening Tendency', 'scaling': 1e3, 'unit': 'g/kg/s', 'vmax': 1.2e-5, 'vmin': -1.2e-5},
    'DQ2PHYS': {'var_title': 'Liquid Tendency', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 0.0015, 'vmin': -0.0015},
    'DQ3PHYS': {'var_title': 'Ice Tendency', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': 0.0015, 'vmin': -0.0015},
    'DQnPHYS': {'var_title': 'Liquid + Ice Tendency', 'scaling': 1e6, 'unit': 'mg/kg/s', 'vmax': .0015, 'vmin': -.0015},
    'DUPHYS': {'var_title': 'Zonal Wind Tendency', 'scaling': 1., 'unit': 'm/s²', 'vmax': 2.2e-6, 'vmin': -2.2e-6}
}

# Map model name -> full path to the E3SM run directory (contains *.eam.h1.*.nc files)
online_paths = {
    'joint': '/pscratch/sd/k/kfrields/climsim-online-data/scratch/5_hour_online_test_3/run',
    'unet':  '/pscratch/sd/k/kfrields/climsim-online-data/scratch/3_day_unet_test_2/run',
}

#seeds = ['seed_7', 'seed_43', 'seed_1024']
#seed_numbers = [7, 43, 1024]

climsim3_figures_save_path_offline = '/global/homes/k/kfrields/climsim-kaggle-edition/figures/offline'
climsim3_figures_save_path_online = '/global/homes/k/kfrields/climsim-kaggle-edition/figures/online'

def ls(data_path = ""):
    return os.popen(" ".join(["ls", data_path])).read().splitlines()

def offline_area_time_mean_3d(arr):
    arr_zonal_mean = data_v2_rh_mc.zonal_bin_weight_3d(arr)
    arr_zonal_time_mean = arr_zonal_mean.mean(axis = 0)
    arr_zonal_time_mean = xr.DataArray(arr_zonal_time_mean.T, dims = ['hybrid pressure (hPa)', 'latitude'], coords = {'hybrid pressure (hPa)':level, 'latitude': lat_bin_mids})
    return arr_zonal_time_mean

def online_area_time_mean_3d(ds, var):
    arr = ds[var].values[1:,:,:]
    arr_reshaped = np.transpose(arr, (0,2,1))
    arr_zonal_mean = data_v2_rh_mc.zonal_bin_weight_3d(arr_reshaped)
    arr_zonal_time_mean = arr_zonal_mean.mean(axis = 0)
    arr_zonal_time_mean = xr.DataArray(arr_zonal_time_mean.T, dims = ['hybrid pressure (hPa)', 'latitude'], coords = {'hybrid pressure (hPa)':level, 'latitude': lat_bin_mids})
    return arr_zonal_time_mean

def difference(ds1, ds2, var):
    arr1 = np.transpose(ds1[var].values[1:,:,:], (0,2,1))  # (time, ncol, lev)
    arr2 = np.transpose(ds2[var].values[1:,:,:], (0,2,1))
    ratio_arr = np.abs(arr1 - arr2)# / (np.abs(arr2) + 1e-30)             # (time, ncol, lev)
    ratio_zm = data_v2_rh_mc.zonal_bin_weight_3d(ratio_arr).mean(axis=0)  # (n_lat_bins, lev)
    return xr.DataArray(ratio_zm.T, dims=['hybrid pressure (hPa)', 'latitude'],
                        coords={'hybrid pressure (hPa)': level, 'latitude': lat_bin_mids})

def area_mean(ds, var):
    arr = ds[var].values
    arr_reshaped = np.transpose(arr, (0,2,1))
    arr_zonal_mean = data_v2_rh_mc.zonal_bin_weight_3d(arr_reshaped)
    return arr_zonal_mean

def zonal_diff(ds_sp, ds_nn, var):
    diff_zonal_mean = (area_mean(ds_nn, var) - area_mean(ds_sp, var)).mean(axis = 0)
    diff_zonal = xr.DataArray(diff_zonal_mean.T, dims = ['level', 'lat'], coords = {'level':level, 'lat': lat_bin_mids})
    return diff_zonal

def get_dp(ds):
    ps = ds['PS']
    p_interface = (ds['hyai'] * ds['P0'] + ds['hybi'] * ds['PS']).values
    if p_interface.shape[0] == 61:
        p_interface = np.swapaxes(p_interface, 0, 1)
    dp = p_interface[:,1:61,:] - p_interface[:,0:60,:]
    return dp

def get_tcp_mean(ds, area_weight):
    cld = ds['TOTCLD'].values
    dp = get_dp(ds)
    tcp = np.sum(cld*dp, axis = 1)/9.81
    tcp_mean = np.average(tcp, weights = area_weight, axis = 1)
    return tcp_mean
'''
def read_mmf_online_data(num_years):
    assert num_years <= 5 and num_years >= 1
    years_regexp = '34567'[:num_years]
    ds_mmf_1 = xr.open_mfdataset(f'/pscratch/sd/z/zeyuanhu/hu_etal2024_data_v2/data/h0/5year/mmf_ref/control_fullysp_jan_wmlio_r3.eam.h0.000[{years_regexp}]*.nc')
    #ds_mmf_1 = xr.open_mfdataset(f'/pscratch/sd/z/zeyuanhu/hu_etal2024_data_v2/data/h0/5year/mmf_ref/control_fullysp_jan_wmlio_r3.eam.h0.000[{years_regexp}]*.nc')
    #ds_mmf_2 = xr.open_mfdataset(f'/pscratch/sd/z/zeyuanhu/hu_etal2024_data_v2/data/h0/5year/mmf_b/control_fullysp_jan_wmlio_r3_b.eam.h0.000[{years_regexp}]*.nc')
    ds_mmf_1['DQnPHYS'] = ds_mmf_1['DQ2PHYS'] + ds_mmf_1['DQ3PHYS']
    ds_mmf_1['PRECT'] = ds_mmf_1['PRECC'] + ds_mmf_1['PRECL']
    ds_mmf_1['TOTCLD'] = ds_mmf_1['CLDICE'] + ds_mmf_1['CLDLIQ']
    
    ds_mmf_2['TOTCLD'] = ds_mmf_2['CLDICE'] + ds_mmf_2['CLDLIQ']
    ds_mmf_2['DQnPHYS'] = ds_mmf_2['DQ2PHYS'] + ds_mmf_2['DQ3PHYS']
    ds_mmf_2['PRECT'] = ds_mmf_2['PRECC'] + ds_mmf_2['PRECL']
    return ds_mmf_1#, ds_mmf_2
'''

def read_new_mmf_h1_data(run_dir, num_days=None):
    """Load daily h1 output from a new MMF test run (e.g. new_mmf_test_run_4).

    Args:
        run_dir:  Path to the E3SM run directory containing h1 .nc files.
        num_days: If set, only load the first num_days files (one file = one day).
    """
    h1_files = sorted(glob.glob(os.path.join(run_dir, '*.eam.h1.*.nc')))
    if len(h1_files) == 0:
        print('No h1 files found in {}'.format(run_dir))
        return None
    if num_days is not None:
        h1_files = h1_files[:num_days]
    ds = xr.open_mfdataset(h1_files, chunks=None)
    ds['DQnPHYS'] = ds['DQ2PHYS'] + ds['DQ3PHYS']
    ds['TOTCLD']  = ds['CLDLIQ']  + ds['CLDICE']
    if 'PRECT' not in ds:
        ds['PRECT'] = ds['PRECC'] + ds['PRECL']
    return ds

def read_nn_online_data(model_name):
    run_dir = online_paths[model_name]
    h1_files = sorted(glob.glob(os.path.join(run_dir, '*.eam.h1.*.nc')))
    if len(h1_files) == 0:
        print('No h1 files found for {} at {}'.format(model_name, run_dir))
        return None
    ds_nn = xr.open_mfdataset(h1_files, chunks=None)
    ds_nn['DQnPHYS'] = ds_nn['DQ2PHYS'] + ds_nn['DQ3PHYS']
    ds_nn['TOTCLD']  = ds_nn['CLDICE']  + ds_nn['CLDLIQ']
    if 'PRECT' not in ds_nn:
        ds_nn['PRECT'] = ds_nn['PRECC'] + ds_nn['PRECL']
    return ds_nn

def get_pressure_area_weights(ds, surface_type = None):
    ds_dp = get_dp(ds)
    ds_total_weight = ds_dp * area_weight[None, None, :]
    ds_total_weight = ds_total_weight.mean(axis = 0)
    ds_total_weight = ds_total_weight/ds_total_weight.sum()
    if surface_type is None:
        return ds_total_weight
    elif surface_type == 'land':
        land_area = ds['LANDFRAC'].values * grid_area[None, :]
        land_area_sums = np.array([[np.sum(land_area[t,:][data_v2_rh_mc.lat_bin_dict[lat_bin]]) for lat_bin in data_v2_rh_mc.lat_bin_dict.keys()] for t in range(land_area.shape[0])])
        land_area_divs = np.stack([np.divide(1, land_area_sums[:, bin_index], where=~(land_area_sums[:, bin_index] == 0), out=np.zeros_like(land_area_sums[:, bin_index])) for bin_index in data_v2_rh_mc.lat_bin_indices], axis=1)
        land_area_weighting = land_area * land_area_divs
        return land_area_weighting
    elif surface_type == 'ocean':
        ocean_area = ds['OCNFRAC'].values * grid_area[None, :]
        ocean_area_sums = np.array([[np.sum(ocean_area[t,:][data_v2_rh_mc.lat_bin_dict[lat_bin]]) for lat_bin in data_v2_rh_mc.lat_bin_dict.keys()] for t in range(ocean_area.shape[0])])
        ocean_area_divs = np.stack([np.divide(1, ocean_area_sums[:, bin_index], where=~(ocean_area_sums[:, bin_index] == 0), out=np.zeros_like(ocean_area_sums[:, bin_index])) for bin_index in data_v2_rh_mc.lat_bin_indices], axis=1)
        ocean_area_weighting = ocean_area * ocean_area_divs
        return ocean_area_weighting
    elif surface_type == 'ice':
        ice_area = ds['ICEFRAC'].values * grid_area[None, :]
        ice_area_sums = np.array([[np.sum(ice_area[t,:][data_v2_rh_mc.lat_bin_dict[lat_bin]]) for lat_bin in data_v2_rh_mc.lat_bin_dict.keys()] for t in range(ice_area.shape[0])])
        ice_area_divs = np.stack([np.divide(1, ice_area_sums[:, bin_index], where=~(ice_area_sums[:, bin_index] == 0), out=np.zeros_like(ice_area_sums[:, bin_index])) for bin_index in data_v2_rh_mc.lat_bin_indices], axis=1)
        ice_area_weighting = ice_area * ice_area_divs
        return ice_area_weighting
    else:
        raise ValueError("Invalid surface type. Choose from 'land', 'ocean', or 'ice'.")

def plot_online_zonal_mean_bias_model_comparison(var, ds_mmf, ds_nn, num_days, show=True, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 4), constrained_layout=True)
    labels = ['({})'.format(letter) for letter in string.ascii_lowercase[:2]]
    latitude_ticks = [-60, -30, 0, 30, 60]
    latitude_labels = ['60S', '30S', '0', '30N', '60N']

    zonal_mean_bias = {model: online_var_settings[var]['scaling'] * (online_area_time_mean_3d(ds_nn[model], var) - online_area_time_mean_3d(ds_mmf, var)) for model in model_names}

    joint_bias = zonal_mean_bias['joint'].plot(ax=axs[0], add_colorbar=False, cmap='RdBu_r', vmin=online_var_settings[var]['vmin'], vmax=online_var_settings[var]['vmax'])
    axs[0].set_title('{} {}'.format(labels[0], model_names['joint']))
    axs[0].invert_yaxis()
    axs[0].set_xlabel('Latitude')
    axs[0].set_ylabel("Hybrid pressure (hPa)")
    fig.colorbar(joint_bias, ax=axs[0])

    unet_bias = zonal_mean_bias['unet'].plot(ax=axs[1], add_colorbar=False, cmap='RdBu_r', vmin=online_var_settings[var]['vmin'], vmax=online_var_settings[var]['vmax'])
    axs[1].set_title('{} {}'.format(labels[1], model_names['unet']))
    axs[1].invert_yaxis()
    axs[1].set_xlabel('Latitude')
    axs[1].set_ylabel('')
    fig.colorbar(unet_bias, ax=axs[1])

    if var == 'CLDICE':
        for ax in axs:
            ax.plot(lat_bin_mids, idx_tropopause_zm, 'k--')

    for ax in axs:
        ax.set_xticks(latitude_ticks)
        ax.set_xticklabels(latitude_labels)

    plt.suptitle('{}-day {} ({}) zonal mean bias (Joint & U-Net vs MMF)'.format(num_days, online_var_settings[var]['var_title'], online_var_settings[var]['unit']), fontsize=14)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'online_{}_day_zonal_mean_{}_bias_model_comparison.png'.format(num_days, var)), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def h1_zonal_time_mean(ds_run, var):
    """Area-weighted zonal mean over all h1 timesteps (no spinup skip)."""
    arr = ds_run[var].values          # (time, lev, ncol)
    arr = np.transpose(arr, (0, 2, 1))  # -> (time, ncol, lev)
    arr_zm = data_v2_rh_mc.zonal_bin_weight_3d(arr)
    arr_mean = arr_zm.mean(axis=0)
    return xr.DataArray(arr_mean.T,
                        dims=['hybrid pressure (hPa)', 'latitude'],
                        coords={'hybrid pressure (hPa)': level, 'latitude': lat_bin_mids})

def compute_rmse_zonal(ds_run, ds_ref, var, scaling=1.0):
    """Compute RMSE per column then take the area-weighted zonal average.

    Returns an xr.DataArray with dims ('hybrid pressure (hPa)', 'latitude').
    """
    n_time = min(ds_run[var].shape[0], ds_ref[var].shape[0]) - 1

    y_pred = np.transpose(ds_run[var].values[1:n_time + 1], (0, 2, 1))  # (time, ncol, lev)
    y_true = np.transpose(ds_ref[var].values[1:n_time + 1], (0, 2, 1))  # (time, ncol, lev)

    rmse = scaling * np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))  # (ncol, lev)

    # area-weighted zonal average of per-column RMSE -> (n_lat_bins, lev)
    rmse_zm = data_v2_rh_mc.zonal_bin_weight_3d(rmse[np.newaxis])[0]

    return xr.DataArray(rmse_zm.T,
                        dims=['hybrid pressure (hPa)', 'latitude'],
                        coords={'hybrid pressure (hPa)': level, 'latitude': lat_bin_mids})


def plot_r2_comparison(ds_mmf, ds_nn, num_days, vars_to_plot=None, show=True, save_path=None):
    """Plot RMSE (unet vs MMF) and (joint-unet)/unet prediction ratio as a grid: one row per variable, two columns.

    Args:
        ds_mmf:        Reference MMF xarray Dataset.
        ds_nn:         Dict mapping model key -> xarray Dataset.
        num_days:      Number of days used (for title/filename).
        vars_to_plot:  List of variable names to include (defaults to all in online_var_settings).
        show:          Display the figure interactively.
        save_path:     Directory to save the figure (skipped if None).
    """
    if vars_to_plot is None:
        vars_to_plot = list(online_var_settings.keys())

    n_vars = len(vars_to_plot)
    latitude_ticks  = [-60, -30, 0, 30, 60]
    latitude_labels = ['60S', '30S', '0', '30N', '60N']

    fig, axs = plt.subplots(n_vars, 2, figsize=(11, 4 * n_vars), constrained_layout=True)
    if n_vars == 1:
        axs = axs[np.newaxis, :]

    panel_labels = ['({})'.format(letter) for letter in string.ascii_lowercase[:2 * n_vars]]

    for row_idx, var in enumerate(vars_to_plot):
        settings = online_var_settings[var]

        rmse_unet = compute_rmse_zonal(ds_nn['unet'], ds_mmf, var, scaling=settings['scaling'])

        # zonal time-mean of the actual predictions for each model
        #zm_joint = online_area_time_mean_3d(ds_nn['joint'], var)
        #zm_unet  = online_area_time_mean_3d(ds_nn['unet'],  var)
        pred_ratio_da = difference(ds_nn['joint'], ds_nn['unet'], var)

        # --- left: deterministic (unet) RMSE ---
        ax0 = axs[row_idx, 0]
        im0 = rmse_unet.plot(ax=ax0, add_colorbar=False, cmap='viridis')
        fig.colorbar(im0, ax=ax0, label='RMSE ({})'.format(settings['unit']))
        ax0.set_title('{} {} RMSE — {}'.format(panel_labels[row_idx * 2], model_names['unet'], settings['var_title']))
        ax0.invert_yaxis()
        ax0.set_xlabel('Latitude')
        ax0.set_ylabel('Hybrid pressure (hPa)')
        ax0.set_xticks(latitude_ticks)
        ax0.set_xticklabels(latitude_labels)

        # --- right: |joint predictions| / |unet predictions| ---
        ax1 = axs[row_idx, 1]
        
        troposphere_vals = pred_ratio_da.values[:, 12:]  # exclude stratosphere (levels 0-11)
        vmin1 = max(np.nanpercentile(troposphere_vals, 20), 1e-9)  # LogNorm requires vmin > 0
        vmax1 = np.nanpercentile(troposphere_vals, 95)
        im1 = pred_ratio_da.plot(ax=ax1, add_colorbar=False, cmap='YlOrRd',
                                 norm=LogNorm(vmin=vmin1, vmax=vmax1))
        fig.colorbar(im1, ax=ax1, label='|Predicted Residual|')
        ax1.set_title('{} |Predicted Residual| — {}'.format(panel_labels[row_idx * 2 + 1], settings['var_title']))
        ax1.invert_yaxis()
        ax1.set_xlabel('Latitude')
        ax1.set_ylabel('')
        ax1.set_xticks(latitude_ticks)
        ax1.set_xticklabels(latitude_labels)

        for ax in [ax0, ax1]:
            if var == 'CLDICE':
                ax.plot(lat_bin_mids, idx_tropopause_zm, 'k--', label='Tropopause')
                ax.legend(fontsize=8)

    plt.suptitle('{}-day: Deterministic U-Net RMSE | |Predicted Residual|'.format(num_days), fontsize=14)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = 'online_{}_day_r2_comparison.png'.format(num_days)
        plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
        print('Saved: {}'.format(fname))
    if show:
        plt.show()
    else:
        plt.close()


def plot_single_run_zonal_mean(run_dir, var, run_name=None, ref_ds=None, show=True, save_path=None):
    """Plot zonal mean (or bias vs reference) for a single run from h1 daily output.

    Args:
        run_dir:   Path to the E3SM run directory containing h1 .nc files.
        var:       Variable name (must be in online_var_settings).
        run_name:  Label for the plot title (defaults to basename of run_dir).
        ref_ds:    Optional reference xarray Dataset. If provided plots bias (run - ref).
        show:      Display the figure interactively.
        save_path: Directory to save the figure (skipped if None).
    """
    if run_name is None:
        run_name = os.path.basename(run_dir.rstrip('/'))

    h1_files = sorted(glob.glob(os.path.join(run_dir, '*.eam.h1.*.nc')))
    if len(h1_files) == 0:
        print('No h1 files found in {}'.format(run_dir))
        return
    ds_run = xr.open_mfdataset(h1_files)
    ds_run['DQnPHYS'] = ds_run['DQ2PHYS'] + ds_run['DQ3PHYS']
    ds_run['TOTCLD']  = ds_run['CLDLIQ']  + ds_run['CLDICE']
    n_days = len(ds_run.time)

    settings = online_var_settings[var]

    if ref_ds is not None:
        run_mean = h1_zonal_time_mean(ds_run, var)
        ref_mean = online_area_time_mean_3d(ref_ds, var)
        plot_data = settings['scaling'] * (run_mean - ref_mean)
        cmap = 'RdBu_r'
        vmin, vmax = settings['vmin'], settings['vmax']
        title_suffix = 'Bias vs. Reference ({}-day mean)'.format(n_days)
    else:
        plot_data = settings['scaling'] * h1_zonal_time_mean(ds_run, var)
        cmap = 'viridis'
        vmin, vmax = None, None
        title_suffix = 'Zonal Mean ({}-day mean)'.format(n_days)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = plot_data.plot(ax=ax, add_colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=f"{settings['var_title']} ({settings['unit']})")
    ax.invert_yaxis()
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Hybrid pressure (hPa)')

    latitude_ticks  = [-60, -30, 0, 30, 60]
    latitude_labels = ['60S', '30S', '0', '30N', '60N']
    ax.set_xticks(latitude_ticks)
    ax.set_xticklabels(latitude_labels)

    if var == 'CLDICE':
        ax.plot(lat_bin_mids, idx_tropopause_zm, 'k--', label='Tropopause')
        ax.legend(fontsize=8)

    plt.suptitle(f"{run_name}\n{settings['var_title']} {title_suffix}", fontsize=12)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = f'zonal_mean_{var}_{run_name}.png'
        plt.savefig(os.path.join(save_path, fname), dpi=150, bbox_inches='tight')
        print(f'Saved: {fname}')
    if show:
        plt.show()
    else:
        plt.close()


# --- Model vs MMF bias comparison ---#new_mmf_test_run_4
mmf_run_dir = '/pscratch/sd/k/kfrields/climsim-online-data/scratch/new_mmf_no_stratosphere/run'
num_days = 3

ds_mmf = read_new_mmf_h1_data(mmf_run_dir, num_days)
ds_nn = {
    'joint': read_nn_online_data('joint'),
    'unet':  read_nn_online_data('unet'),
}

if ds_mmf is not None and ds_nn['joint'] is not None and ds_nn['unet'] is not None:
    for online_var in tqdm(online_var_settings.keys()):
        plot_online_zonal_mean_bias_model_comparison(
            var=online_var,
            ds_mmf=ds_mmf,
            ds_nn=ds_nn,
            num_days=num_days,
            show=False,
            save_path=os.path.join(climsim3_figures_save_path_online, 'bias_model_comparison')
        )

    plot_r2_comparison(
        ds_mmf=ds_mmf,
        ds_nn=ds_nn,
        num_days=num_days,
        show=False,
        save_path=os.path.join(climsim3_figures_save_path_online, 'r2_comparison')
    )
import numpy as np
from sklearn.metrics import r2_score
import torch
import os, gc
import modulus
from tqdm import tqdm
import sys
from climsim_utils.data_utils import *

grid_path = '/global/cfs/cdirs/m4334/jerry/climsim3_dev/grid_info/ClimSim_low-res_grid-info.nc'

input_mean_v2_rh_mc_file = 'input_mean_v2_rh_mc_pervar.nc'
input_max_v2_rh_mc_file = 'input_max_v2_rh_mc_pervar.nc'
input_min_v2_rh_mc_file = 'input_min_v2_rh_mc_pervar.nc'
output_scale_v2_rh_mc_file = 'output_scale_std_lowerthred_v2_rh_mc.nc'


lbd_qn_file = 'qn_exp_lambda_large.txt'

grid_info = xr.open_dataset(grid_path)

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


input_sub_v2_rh_mc, input_div_v2_rh_mc, out_scale_v2_rh_mc = data_v2_rh_mc.save_norm(write=False) # this extracts only the relevant variables
input_sub_v2_rh_mc = input_sub_v2_rh_mc[None, :]
input_div_v2_rh_mc = input_div_v2_rh_mc[None, :]
out_scale_v2_rh_mc = out_scale_v2_rh_mc[None, :]

lat = grid_info['lat'].values
lon = grid_info['lon'].values
lat_bin_mids = data_v2_rh_mc.lat_bin_mids

v2_rh_mc_input_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_input.npy'
v2_rh_mc_target_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/test_target.npy'
standard_save_path = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/diffusion_models/diff_test/offline/'


def apply_temperature_rules(T):
    # Create an output tensor, initialized to zero
    output = np.zeros_like(T)

    # Apply the linear transition within the range 253.16 to 273.16
    mask = (T >= 253.16) & (T <= 273.16)
    output[mask] = (T[mask] - 253.16) / 20.0  # 20.0 is the range (273.16 - 253.16)

    # Values where T > 273.16 set to 1
    output[T > 273.16] = 1

    # Values where T < 253.16 are already set to 0 by the initialization
    return output

def preprocessing_v2_rh_mc(data, input_path, target_path, input_sub, input_div, lbd_qn, out_scale):
    npy_input = np.load(input_path)
    npy_target = np.load(target_path)

    surface_pressure = npy_input[:, data.ps_index]
    
    hyam_component = (data.hyam * data.p0)[np.newaxis,:]
    hybm_component = data.hybm[np.newaxis,:] * surface_pressure[:, np.newaxis]
    
    pressures = hyam_component + hybm_component
    pressures = pressures.reshape(-1,384,60)
    
    pressures_binned = data.zonal_bin_weight_3d(pressures)
    
    actual_input = npy_input.copy().reshape(-1, data.num_latlon, data.input_feature_len)

    npy_input[:,120:180] = 1 - np.exp(-npy_input[:,120:180] * lbd_qn)
    npy_input = (npy_input - input_sub)/input_div
    npy_input = np.where(np.isnan(npy_input), 0, npy_input)
    npy_input = np.where(np.isinf(npy_input), 0, npy_input)
    npy_input[:,120:120+15] = 0
    npy_input[:,60:120] = np.clip(npy_input[:,60:120], 0, 1.2)
    torch_input = torch.tensor(npy_input).float()

    reshaped_target = npy_target.reshape(-1, data.num_latlon, data.target_feature_len)

    t_before = actual_input[:, :, 0:60]
    qn_before = actual_input[:, :, 120:180]
    liq_frac_before = apply_temperature_rules(t_before)
    qc_before = liq_frac_before * qn_before
    qi_before = (1 - liq_frac_before) * qn_before

    t_new = t_before + reshaped_target[:, :, 0:60]*1200
    qn_new = qn_before + reshaped_target[:, :, 120:180]*1200
    liq_frac_new = apply_temperature_rules(t_new)
    qc_new = liq_frac_new * qn_new
    qi_new = (1 - liq_frac_new) * qn_new
    
    actual_target = np.concatenate((reshaped_target[:, :, 0:120], 
                                    (qc_new - qc_before)/1200, 
                                    (qi_new - qi_before)/1200, 
                                    reshaped_target[:, :, 180:240], 
                                    reshaped_target[:, :, 240:]), axis=2)
    return torch_input, actual_input, actual_target, pressures_binned

def preprocessing_v6(data, input_path, target_path, input_sub, input_div, lbd_qn, out_scale):
    npy_input = np.load(input_path)
    npy_target = np.load(target_path)
    
    surface_pressure = npy_input[:, data.ps_index]
    
    hyam_component = (data.hyam * data.p0)[np.newaxis,:]
    hybm_component = data.hybm[np.newaxis,:] * surface_pressure[:, np.newaxis]
    
    pressures = hyam_component + hybm_component
    pressures = pressures.reshape(-1,384,60)
    
    pressures_binned = data.zonal_bin_weight_3d(pressures)
    
    actual_input = npy_input.copy().reshape(-1, data.num_latlon, data.input_feature_len)

    npy_input[:,120:180] = 1 - np.exp(-npy_input[:,120:180] * lbd_qn)
    npy_input = (npy_input - input_sub)/input_div
    npy_input = np.where(np.isnan(npy_input), 0, npy_input)
    npy_input = np.where(np.isinf(npy_input), 0, npy_input)
    npy_input[:,120:120+15] = 0
    npy_input[:,60:120] = np.clip(npy_input[:,60:120], 0, 1.2)
    torch_input = torch.tensor(npy_input).float()

    reshaped_target = npy_target.reshape(-1, data.num_latlon, data.target_feature_len)

    t_before = actual_input[:, :, 0:60]
    qn_before = actual_input[:, :, 120:180]
    liq_frac_before = apply_temperature_rules(t_before)
    qc_before = liq_frac_before * qn_before
    qi_before = (1 - liq_frac_before) * qn_before

    t_new = t_before + reshaped_target[:, :, 0:60]*1200
    qn_new = qn_before + reshaped_target[:, :, 120:180]*1200
    liq_frac_new = apply_temperature_rules(t_new)
    qc_new = liq_frac_new * qn_new
    qi_new = (1 - liq_frac_new) * qn_new
    
    actual_target = np.concatenate((reshaped_target[:, :, 0:120], 
                                    (qc_new - qc_before)/1200, 
                                    (qi_new - qi_before)/1200, 
                                    reshaped_target[:, :, 180:240], 
                                    reshaped_target[:, :, 240:]), axis=2)
    return torch_input, actual_input, actual_target, pressures_binned

# standard models
standard_unet_seed_7_path = '/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/diffusion_models/diff_test/unet_model.pt'
standard_model_paths = {
    'standard_unet_seed_7': standard_unet_seed_7_path
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference_model(data, model_path, actual_input, torch_input, out_scale):
    model = torch.jit.load(model_path).to(device)
    model.eval()
    batch_pred_list = []
    batch_size = data.num_latlon
    with torch.no_grad():
        for i in tqdm(range(0, torch_input.shape[0], batch_size)):
            batch = torch_input[i:i+batch_size].to(device)
            batch_pred = model(batch)
            batch_pred[:, 60:75] = 0
            batch_pred[:, 120:135] = 0
            batch_pred[:, 180:195] = 0
            batch_pred[:, 240:255] = 0
            batch_pred_list.append(batch_pred.cpu().numpy() / out_scale)
    model_preds = np.stack(batch_pred_list, axis=0)

    t_before = actual_input[:, :, 0:60]
    qn_before = actual_input[:, :, 120:180]
    liq_frac_before = apply_temperature_rules(t_before)
    qc_before = liq_frac_before * qn_before
    qi_before = (1 - liq_frac_before) * qn_before

    t_new = t_before + model_preds[:, :, 0:60]*1200
    qn_new = qn_before + model_preds[:, :, 120:180]*1200
    liq_frac_new = apply_temperature_rules(t_new)
    qc_new = liq_frac_new * qn_new
    qi_new = (1 - liq_frac_new) * qn_new
    
    actual_preds = np.concatenate((model_preds[:, :, 0:120], 
                                  (qc_new - qc_before)/1200, 
                                  (qi_new - qi_before)/1200, 
                                   model_preds[:, :, 180:240], 
                                   model_preds[:, :, 240:]), axis=2)

    del model
    del batch_pred_list
    gc.collect()
    torch.cuda.empty_cache()
    return actual_preds

def inference_model_conf_loss(data, model_path, actual_input, torch_input, out_scale):
    model = torch.jit.load(model_path).to(device)
    model.eval()
    batch_pred_list = []
    batch_conf_list = []
    batch_size = data.num_latlon
    with torch.no_grad():
        for i in tqdm(range(0, torch_input.shape[0], batch_size)):
            batch = torch_input[i:i+batch_size].to(device)
            batch_pred, batch_conf = model(batch)
            batch_pred[:, 60:75] = 0
            batch_pred[:, 120:135] = 0
            batch_pred[:, 180:195] = 0
            batch_pred[:, 240:255] = 0
            batch_pred_list.append(batch_pred.cpu().numpy() / out_scale)
            batch_conf_list.append(batch_conf.cpu().numpy())
    model_preds = np.stack(batch_pred_list, axis=0)
    model_conf = np.stack(batch_conf_list, axis=0)

    t_before = actual_input[:, :, 0:60]
    qn_before = actual_input[:, :, 120:180]
    liq_frac_before = apply_temperature_rules(t_before)
    qc_before = liq_frac_before * qn_before
    qi_before = (1 - liq_frac_before) * qn_before

    t_new = t_before + model_preds[:, :, 0:60]*1200
    qn_new = qn_before + model_preds[:, :, 120:180]*1200
    liq_frac_new = apply_temperature_rules(t_new)
    qc_new = liq_frac_new * qn_new
    qi_new = (1 - liq_frac_new) * qn_new
    
    actual_preds = np.concatenate((model_preds[:, :, 0:120], 
                                  (qc_new - qc_before)/1200, 
                                  (qi_new - qi_before)/1200, 
                                   model_preds[:, :, 180:240], 
                                   model_preds[:, :, 240:]), axis=2)

    del model
    del batch_pred_list
    del batch_conf_list
    gc.collect()
    torch.cuda.empty_cache()
    return actual_preds, model_conf

torch_input_v2_rh_mc, actual_input_v2_rh_mc, actual_target, pressures_binned = preprocessing_v2_rh_mc(data = data_v2_rh_mc, 
                                                                                                      input_path = v2_rh_mc_input_path, 
                                                                                                      target_path = v2_rh_mc_target_path, 
                                                                                                      input_sub = input_sub_v2_rh_mc, 
                                                                                                      input_div = input_div_v2_rh_mc, 
                                                                                                      lbd_qn = lbd_qn, 
                                                                                                      out_scale = out_scale_v2_rh_mc)

np.save('/pscratch/sd/k/kfrields/hugging/E3SM-MMF-preprocessing/v2_rh_mc/test_set/actual_input.npy', actual_input_v2_rh_mc)
np.save('/pscratch/sd/k/kfrields/hugging/E3SM-MMF-preprocessing/v2_rh_mc/test_set/actual_target.npy', actual_target)

# standard u-net
print("Running standard u-net inference...")
standard_unet_preds_1 = inference_model(data_v2_rh_mc,
                                        standard_model_paths['standard_unet_seed_7'],
                                        actual_input_v2_rh_mc,
                                        torch_input_v2_rh_mc,
                                        out_scale_v2_rh_mc)
seeds = [7]

np.savez(os.path.join(standard_save_path, 'standard_unet_preds.npz'), 
         seed_7 = standard_unet_preds_1)


gc.collect()


print('Finished inferencing all models!')

def show_r2(target, preds):
    assert target.shape == preds.shape, f'target shape {target.shape} does not match preds shape {preds.shape}'
    new_shape = (np.prod(target.shape[:-1]), target.shape[-1])
    target_flattened = target.reshape(new_shape)
    preds_flattened = preds.reshape(new_shape)
    r2_scores = np.array([r2_score(target_flattened[:, i], preds_flattened[:, i]) for i in range(368)])
    r2_scores_capped = r2_scores.copy()
    r2_scores_capped[r2_scores_capped < 0] = 0
    return r2_scores, r2_scores_capped

print('Calculating standard r2')
standard_unet_r2 = {seed: show_r2(actual_target, standard_unet_preds_1) for seed in seeds}

with open(os.path.join(standard_save_path, "standard_unet_r2.pkl"), "wb") as f:
    pickle.dump(standard_unet_r2, f)

def get_coeff(target, pred):
    rss = np.sum((pred - target)**2, axis = 0)
    tss = np.sum((target - np.mean(target, axis = 0)[None,:,:])**2, axis = 0)
    coeff = 1 - rss/tss
    mask = tss == 0
    coeff[mask] = 1.0 * (rss[mask] == 0) 
    return coeff

seed =7
standard_unet_zonal_dTdt_r2 = {seed: get_coeff(actual_target[:,:,:60], standard_unet_preds_1[:,:,:60])}
standard_unet_zonal_dQvdt_r2 = {seed: get_coeff(actual_target[:,:,60:120], standard_unet_preds_1[:,:,60:120]) }
standard_unet_zonal_dQldt_r2 = {seed: get_coeff(actual_target[:,:,120:180], standard_unet_preds_1[:,:,120:180]) }
standard_unet_zonal_dQidt_r2 = {seed: get_coeff(actual_target[:,:,180:240], standard_unet_preds_1[:,:,180:240]) }
standard_unet_zonal_dUdt_r2 = {seed: get_coeff(actual_target[:,:,240:300], standard_unet_preds_1[:,:,240:300]) }
standard_unet_zonal_dVdt_r2 = {seed: get_coeff(actual_target[:,:,300:360], standard_unet_preds_1[:,:,300:360]) }


with open(os.path.join(standard_save_path, "zonal", "standard_unet_zonal_dTdt_r2.pkl"), "wb") as f:
    pickle.dump(standard_unet_zonal_dTdt_r2, f)
with open(os.path.join(standard_save_path, "zonal", "standard_unet_zonal_dQvdt_r2.pkl"), "wb") as f:
    pickle.dump(standard_unet_zonal_dQvdt_r2, f)
with open(os.path.join(standard_save_path, "zonal", "standard_unet_zonal_dQldt_r2.pkl"), "wb") as f:
    pickle.dump(standard_unet_zonal_dQldt_r2, f)
with open(os.path.join(standard_save_path, "zonal", "standard_unet_zonal_dQidt_r2.pkl"), "wb") as f:
    pickle.dump(standard_unet_zonal_dQidt_r2, f)
with open(os.path.join(standard_save_path, "zonal", "standard_unet_zonal_dUdt_r2.pkl"), "wb") as f:
    pickle.dump(standard_unet_zonal_dUdt_r2, f)
with open(os.path.join(standard_save_path, "zonal", "standard_unet_zonal_dVdt_r2.pkl"), "wb") as f:
    pickle.dump(standard_unet_zonal_dVdt_r2, f)
    
del standard_unet_preds_1

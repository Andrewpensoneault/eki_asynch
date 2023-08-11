#!/usr/bin/python
import sys
import shutil
import numpy as np
import os

from tqdm import tqdm
from utils import process_json, get_ids, get_subwatershed
from io_ifc import create_meas_sav, create_test_rec, create_prm, create_gbl, create_batch_job_file, save_statistics_csv, save_particles
from eki import subsample_data, pert, EnKF_step
from latent import create_latent, transform_latent
from run import run_test
from ifc_usgs_fileorder import file_order, usgs_2_id


def main(json_name, ens):
    # Read json file and get directories and number of steps
    test_dict = process_json(json_name)
    tmp_dir = test_dict['tmp_dir']
    out_dir = test_dict['out_dir']
    step_num = test_dict['steps']

    # Get data file location, idx of locations, and standard deviation parameters
    data_file = test_dict['meas_csv']
    usgs = test_dict['meas_usgs']
    meas_std = test_dict['abs_std_meas']
    rel_meas_std = test_dict['rel_std_meas']
    
    # Remove all temp files and copy json into out dir and tries to make output for csv and pickle outputs
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir,f))
    shutil.copyfile(json_name, out_dir + 'test.json')
    os.makedirs(out_dir + 'csv/', exist_ok=True)
    os.makedirs(out_dir + 'npy/', exist_ok=True)
    
    # Get list of IDs and create all necesary files
    id_list = get_ids(test_dict)
    sparse_parent = get_subwatershed(test_dict, id_list)
    latent_var = create_latent(test_dict, sparse_parent, ens)
    prm_ens, id_list = transform_latent(test_dict, sparse_parent, latent_var)

    # Create all necessary files for running tests
    create_meas_sav(test_dict, id_list)
    create_test_rec(test_dict, id_list)
    create_prm(test_dict, id_list, prm_ens, ens)
    create_gbl(test_dict, ens)
    create_batch_job_file(tmp_dir)
    
    # Get data from csv file and seperate it into EKI / Plotting / IDs and save to file
    data = np.genfromtxt(data_file, delimiter=',', skip_header=True)
    data_tmp = data[:,1:]
    #location in data file where the used id is
    idx_id = np.where(file_order == usgs_2_id[usgs])[0]
    data_use = data_tmp[:,idx_id]
    data_plot, sav_ids = subsample_data(data_tmp, test_dict, id_list, file_order)
    #TDOD: Change this so usgs can be a list of strings, to enable multiple sensors turned on simultaneously
    idx_meas = np.where(sav_ids == usgs_2_id[usgs])[0] 
    save_statistics_csv(test_dict, sparse_parent, data_plot, name='csv/' + "meas")

    # EKI parameters (y = data, X = latent parameter ensemble, R = measurement uncertainty)
    y = np.reshape(data_use,(-1,1)) 
    R = (rel_meas_std * y.reshape(-1))**2 + meas_std**2
    X_post = latent_var

    # Run test
    for i in tqdm(range(step_num)):
        # Perturb previous parameters, run model, get simulation results - Prior
        X_prior = pert(X_post, test_dict, sparse_parent)   
        prm_ens_prior, _ = transform_latent(test_dict, sparse_parent, X_prior)
        create_prm(test_dict, id_list, prm_ens_prior, ens) 
        Y_prior, Y_plot_prior, Y_mean, Y_std, _, _  = run_test(ens, X_prior, tmp_dir, idx_meas)    
        save_particles(test_dict, sparse_parent, X_prior, Y_plot_prior, name='npy/' + str(i) + '_prior')
        save_statistics_csv(test_dict, sparse_parent, Y_mean, Y_std, X_prior, name='csv/' + str(i) + "_prior")
        
        # Run EKI step, rerun model, record simulation results after assimilation - Posterior 
        X_post = EnKF_step(y, X_prior, Y_prior, R, test_dict, i)
        prm_ens_post, _ = transform_latent(test_dict, sparse_parent, X_post)
        create_prm(test_dict, id_list, prm_ens_post, ens)    
        Y_post, Y_plot_post, Y_mean, Y_std, _, _ = run_test(ens, X_post, tmp_dir, idx_meas) 
        save_particles(test_dict, sparse_parent, X_post, Y_plot_post, name='npy/' + str(i) + "_post")
        save_statistics_csv(test_dict, sparse_parent, Y_mean, Y_std, X_post, name='csv/' + str(i) + "_post")
       
    
if __name__ == "__main__": 
   json_name = sys.argv[1]
   ens = int(sys.argv[2])
   main(json_name, ens)

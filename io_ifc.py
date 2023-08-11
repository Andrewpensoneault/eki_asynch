import numpy as np
from latent import transform_latent_sparse
from typing import List, Tuple, Dict, Union
from utils import time_to_epoch

def create_gbl(test_dict: dict, ens: int) -> None:
    """
    Create GBL files based on the given test dictionary.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        ens (int): Number of GBL files to create.

    Returns:
        None
    """
    
    # Times utilized for .gbl
    start_time = test_dict["time_start"]
    end_time = test_dict["time_end"]
    epoch_time_start = int(time_to_epoch(start_time))
    epoch_time_end = int(time_to_epoch(end_time))
    
    # Utilized directories
    rain_dir = test_dict["rain_dir"]
    out_dir = test_dict["out_dir"]
    tmp_dir = test_dict["tmp_dir"]
    
    # Filenames utilized
    rvr_name = test_dict["rvr"]
    mon_name = test_dict["mon"]
    rec_name = tmp_dir + 'init.rec' 
    sav_name = tmp_dir + 'meas.sav' 
    
    # List containing contents of .gbl file
    # For more details, view https://github.com/Iowa-Flood-Center/asynch/tree/develop/examples
    # TODO: Adjust the gbl_list to work with other model numbers
    gbl_list = [
                "609",
                start_time, 
                end_time, 
                "0",
                "1", 
                "State0", 
                "Classic",
                "1 1", 
                "30 10 30", 
                "0 " + rvr_name,
                "0 ",
                "2 " + rec_name,
                "4", 
                "5 " + rain_dir, 
                "10 60 " + str(epoch_time_start) + " " + str(epoch_time_end),
                "7 " + mon_name,  
                str(epoch_time_start) + " " + str(epoch_time_end),
                "0",
                "0",
                "0",
                "0", 
                "2 60 ",
                "0 ", 
                "1 " + sav_name, 
                "0", 
                "0",
                tmp_dir, 
                ".1 10.0 .9", 
                "0", 
                "2", 
                "1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2", 
                "1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2", 
                "1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2", 
                "1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2", 
                "#"]
    # Make a copy of gbl_list to modify for ensemble members
    gbl_list_copy = gbl_list.copy()  
    
    # Write each .gbl within tmp directory
    for i in range(ens):
        gbl_name = tmp_dir + str(i) + ".gbl"
        prm_name = tmp_dir + str(i) + ".prm"
        uini_name = tmp_dir + str(i) + ".rec" 
        csv_name = tmp_dir + str(i) + ".csv" 
        gbl_list_copy[10] = gbl_list[10] + prm_name
        gbl_list_copy[21] = gbl_list[21] + csv_name
        gbl_list_copy[26] = gbl_list[26] + "_" + str(i)
        f = open(gbl_name,'w')
        for item in gbl_list_copy:
            f.write("%s\n" % item)
        f.close()

def create_prm(test_dict: dict, id_list: list, prm_array: np.ndarray, ens: int) -> None:
    """
    Create PRM files based on the given test dictionary, ID list, and PRM array.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        id_list (list): List of IDs.
        prm_array (np.ndarray): Array of PRM values.
        ens (int): Number of PRM files to create.

    Returns:
        None
    """
    
    # Format is:
    # Total number of IDs
    # ID 1
    # Parameters
    # ID 2
    # ...
    
    # Initialize empty list of size total number of rows
    id_num = len(id_list)
    prm_list = [[] for _ in range(2 * id_num + 1)]

    
    tmp_dir = test_dict["tmp_dir"]
    prm_num = prm_array.shape[1]
    prm_list[0] = str(id_num)
    for i in range(ens):
        prm_name = tmp_dir + str(i) + ".prm"
        with open(prm_name, 'w') as f:
            for j in range(id_num):
                prm_list[1 + 2 * j] = str(id_list[j])
                prm_list[2 + 2 * j] = " ".join([str(item) for item in prm_array[:, j, i]])
            for item in prm_list:
                f.write("%s\n" % item)

def create_meas_sav(test_dict: dict, id_list: list) -> None:
    """
    Create a filtered SAV file based on the given test dictionary and ID list for the test.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        id_list (list): List of IDs for filtering the SAV file.

    Returns:
        None
    """
    # Get necessary parameters
    sav_name = test_dict['meas_sav']
    tmp_dir = test_dict['tmp_dir']

    # Read existing SAV file and filter lines based on ID list
    with open(sav_name, 'r') as f:
        sav_lines = [line.strip() for line in f.readlines() if line.strip()]
        new_lines = [line for line in sav_lines if int(line) in id_list]

    # Write the filtered lines to a new SAV file
    sav_name = tmp_dir + "meas.sav"
    with open(sav_name, 'w') as f:
        for line in new_lines:
            f.write("%s\n" % line)
                
def create_test_rec(test_dict: dict, id_list: list) -> None:
    """
    Create a filtered REC file based on the given test dictionary and ID list.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        id_list (list): List of IDs (integers) for filtering the REC file.

    Returns:
        None
    """
    # Get necessary parameters
    rec_name = test_dict['rec']
    tmp_dir = test_dict['tmp_dir']

    # Read existing REC file and filter lines based on ID list
    with open(rec_name, 'r') as f:
        rec_lines = [line.strip() for line in f.readlines() if line.strip()]

    id_num = len(id_list)
    rec_lines[1] = str(id_num)

    new_lines = rec_lines[:3]
    id_lines = rec_lines[3::2]
    state_lines = rec_lines[4::2]

    for i, line in enumerate(id_lines):
        if int(line) in id_list:
            new_lines.append(line)
            new_lines.append(state_lines[i])

    # Write the filtered lines to a new REC file
    rec_name = tmp_dir + "init.rec"
    with open(rec_name, 'w') as f:
        for item in new_lines:
            f.write("%s\n" % item)

def save_statistics_csv(test_dict, sparse_parent, Y_mean, Y_std=None, X_mat=None, name="results"):
    """
    Save statistical results to CSV files.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        sparse_parent (dict): Sparse parent information.
        Y_mean (np.ndarray): Mean results to be saved to a CSV file.
        Y_std (np.ndarray, optional): Standard deviation results to be saved to a CSV file.
        X_mat (np.ndarray, optional): Parameter data to compute mean and standard deviation.
        name (str, optional): Prefix for the output CSV file names.

    Returns:
        None
    """
    # Get necessary parameters
    out_dir = test_dict["out_dir"]
    tmp_dir = test_dict["tmp_dir"]
    sav_name = tmp_dir + "meas.sav" 

    # Load data from SAV file
    sav_val = np.genfromtxt(sav_name, delimiter=',', ndmin=1)
    sav_num = len(sav_val)
    title_y = sav_val.reshape(1, sav_num)

    # Save mean results to CSV
    Y_mean_out_content = np.concatenate((title_y, Y_mean), axis=0)
    out_name_mean = out_dir + str(name) + "_mean.csv"
    np.savetxt(out_name_mean, Y_mean_out_content, delimiter=",", fmt="%.5e")

    # Save standard deviation results to CSV if provided
    if Y_std is not None:
        Y_std_out_content = np.concatenate((title_y, Y_std), axis=0)
        out_name_std = out_dir + str(name) + "_std.csv"
        np.savetxt(out_name_std, Y_std_out_content, delimiter=",", fmt="%.5e")

    # Save parameter mean and standard deviation to CSV if X_mat is provided
    if X_mat is not None:
        X_sparse = transform_latent_sparse(test_dict, sparse_parent, X_mat)
        X_mean = np.mean(X_sparse, axis=1, keepdims=True)
        X_std = np.std(X_sparse, axis=1, keepdims=True)

        X_name_mean = out_dir + str(name) + "_params_mean.csv"
        np.savetxt(X_name_mean, X_mean, delimiter=",", fmt="%.5e")

        X_name_std = out_dir + str(name) + "_params_std.csv"
        np.savetxt(X_name_std, X_std, delimiter=",", fmt="%.5e")

def save_particles(test_dict, sparse_parent, X_particle, Y_particle, name="results"):
    """
    Save particle data to NPY files.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        sparse_parent (dict): Sparse parent information.
        X_particle (np.ndarray): Particle data to be saved to a NPY file.
        Y_particle (np.ndarray): Particle results to be saved to a NPY file.
        name (str, optional): Prefix for the output NPY file names.

    Returns:
        None
    """
    # Get necessary parameters
    out_dir = test_dict["out_dir"]
    tmp_dir = test_dict["tmp_dir"]
    sav_name = tmp_dir + "meas.sav" 

    # Load data from SAV file
    sav_val = np.genfromtxt(sav_name, delimiter=',', ndmin=1)
    sav_num = len(sav_val)
    title_y = sav_val.reshape(1, sav_num)

    # Transform particle latent parameters
    X_sparse = transform_latent_sparse(test_dict, sparse_parent, X_particle)

    # Save particle data to NPY files
    X_particle_name = out_dir + str(name) + '_params_particles.npy'
    Y_particle_name = out_dir + str(name) + '_particles.npy'
    
    with open(Y_particle_name, 'wb') as f:
        np.save(f, Y_particle)
        
    with open(X_particle_name, 'wb') as f:
        np.save(f, X_sparse)

def create_batch_job_file(tmp_dir: str) -> None:
    """
    Create a batch job file for running EKI simulations.

    Args:
        tmp_dir (str): Temporary directory where the batch job file will be created.

    Returns:
        None
    """
    with open(tmp_dir + 'submit_job.job', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#$ -N EKI_job\n')
        f.write('#$ -pe orte 2\n')
        f.write('#$ -q IFC\n')
        f.write('#$ -cwd\n')
        f.write('#$ -o /dev/null\n')
        f.write('#$ -e /dev/null\n')
        f.write('\n')
        f.write('filename=$(($SGE_TASK_ID - 1))\n')
        f.write('mpirun -np 2 asynch ' + tmp_dir + '$filename.gbl\n')
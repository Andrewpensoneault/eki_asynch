import os
import numpy as np
import time
from typing import List, Tuple, Dict, Union

def run_test(ens: int, X: np.ndarray, tmp_dir: str, idx_meas: np.ndarray) -> Tuple[np.ndarray]:
    """
    Run the test with given ensemble size, latent parameter ensemble, temporary directory, and measurement indices.

    Args:
        ens (int): Number of ensemble members.
        X (np.ndarray): Latent parameter ensemble.
        tmp_dir (str): Temporary directory path.
        idx_meas (np.ndarray): Array containing measurement indices.

    Returns:
        Tuple[np.ndarray]: A tuple containing simulation results and statistics.
    """
    
    # Runs test utilizing 'submit_job.job' script, submiting array job
    job = "qsub -t 1:" + str(ens) + ' ' + tmp_dir + 'submit_job.job'
    procs = os.system(job)

    # Tries to read results files, retries every 100 seconds
    while True:
        try:
            read_values = [np.genfromtxt(tmp_dir + str(j) + ".csv", delimiter=',', skip_header=2) for j in range(ens)]
            count_all = np.array([a.size for a in read_values])

            #makes sure results are all the same size and not all 0
            if (np.max(count_all) - np.min(count_all)) == 0:
                if np.any(np.max(count_all) == 0):
                    pass
                else:
                    break
        except:
            print("not finished, waiting 100 seconds")
            time.sleep(100)
            #TODO: probably should ensure this doesnt go forever if unmonitored

    # Removes last column (bug associated with written csv file, extra empty column)
    read_values_fixed = [results[:, :-1] for results in read_values]
    
    # Gets results at measured locations
    read_values_measured = [results[:, idx_meas] for results in read_values_fixed]
    Y = np.concatenate([np.reshape(results, (-1, 1)) for results in read_values_measured], 1)
    
    # Calculates mean, standard deviation, and full list of results at plotting locations
    Y_plot_mean = np.mean(np.array(read_values_fixed), 0)
    Y_plot_std = np.std(np.array(read_values_fixed), axis=0)
    Y_plot = np.array(read_values_fixed)
    
    # Calculates the mean and standard deviation of latent variables
    X_plot_mean = np.mean(X, axis=1, keepdims=True)
    X_plot_std = np.std(X, axis=1, keepdims=True)

    # Remove temporary CSV files
    for j in range(ens):
        os.remove(tmp_dir + str(j) + ".csv")

    return Y, Y_plot, Y_plot_mean, Y_plot_std, X_plot_mean, X_plot_std
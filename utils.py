from dateutil import parser
import json
from scipy.sparse import coo_matrix
from typing import List, Tuple, Dict, Union
import numpy as np

## Utility functions

def process_json(json_name: str) -> dict:
    """
    Read and parse a JSON file.

    Args:
        json_name (str): Name of the JSON file.

    Returns:
        dict: Parsed JSON data as a dictionary.
    """
    with open(json_name) as f:
        test_dict = json.load(f)
    return test_dict

def time_to_epoch(time: str) -> float:
    """
    Convert a time string to epoch timestamp.

    Args:
        time (str): Time string in any valid format.

    Returns:
        float: Epoch timestamp representing the given time.
    """
    return parser.parse(time).timestamp()

def get_ids(test_dict: dict) -> List[int]:
    """
    Get the list of IDs from the PRM file specified in the test dictionary.

    Args:
        test_dict (dict): Test dictionary containing required parameters.

    Returns:
        List[int]: Sorted list of IDs extracted from the PRM file.
    """
    prm_name = test_dict['prm']
    with open(prm_name, 'r') as f:
        prm_lines = [line for line in f.readlines() if line.strip()]
    id_list_prm = [int(i.strip('\n')) for i in prm_lines[1::2]]
    id_list = np.sort(id_list_prm)
    return id_list


def get_subwatershed(test_dict, id_list_use):
    """
    Get a sparse matrix representing the subwatershed based on the given test dictionary and the list of IDs to use.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        id_list_use (List[int]): List of IDs to use for subsetting the subwatershed.

    Returns:
        coo_matrix: Sparse matrix representing the subwatershed.
    """
    # Gets division value
    watershed_csv = test_dict["watershed_csv"]
    watershed_depth = test_dict["watershed_depth"]
    watershed_vals = np.genfromtxt(watershed_csv, delimiter=',', skip_header=True)
    id_subwatershed = watershed_vals[:, 0]
    idx_sort = np.argsort(id_subwatershed)
    id_list = id_subwatershed[idx_sort]

    # Selects the relevent column for index
    if watershed_depth == 4:
        idx_col = 1
    elif watershed_depth == 5:
        idx_col = 2
    elif watershed_depth == 6:
        idx_col = 3
    elif watershed_depth == 7:
        idx_col = 4
    elif watershed_depth == 8:
        idx_col = 5

    id_divs = (watershed_vals[idx_sort, idx_col] - 1).astype(int)

    id_tmp = []
    id_div_tmp = []
    
    # Get only ids in id_list_use
    for i, id_val in enumerate(id_list):
        if id_val in id_list_use:
            id_tmp.append(id_val)
            id_div_tmp.append(id_divs[i])
    id_tmp = np.array(id_tmp)
    id_div_tmp = np.array(id_div_tmp)
    
    # Assigns value from 0 to max to each for divisions, used to eliminate unused indices
    divs_new = 0
    max_div = np.max(id_div_tmp)
    for i in range(max_div + 1):
        count_i = np.sum(id_div_tmp == i)
        if count_i > 0:
            id_div_tmp[id_div_tmp == i] = divs_new
            divs_new += 1

    id_num = len(id_tmp)
    subws_num = len(np.unique(id_div_tmp))
    
    # Create sparse matrix to convert from full parameters to sparse representation
    val_vals = np.ones(id_num)
    col_vals = np.arange(id_num)
    row_vals = id_div_tmp
    sparse_parent = coo_matrix((val_vals, (row_vals, col_vals)), shape=(subws_num, id_num))

    return sparse_parent
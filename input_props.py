import pandas as pd
import numpy as np
from typing import List
import warnings

import input_clean

def unique_data_labels(data_dict: dict):
    list_of_lists = [data_dict[sheet]['labels'] for sheet in data_dict.keys()]
    list_of_labels = []
    for each_list in list_of_lists:
        for each_label in each_list:
            list_of_labels.append(each_label)

    unique_labels = set(list_of_labels)
    
    return unique_labels

def nsamples(data_dict: dict):
    nsamples = 0
    for sheet in data_dict.keys():
        nsamples += data_dict[sheet]['size'][1]

    return nsamples

def nclass(data_dict: dict):
    unique_labels = unique_data_labels(data_dict)
    
    return len(unique_labels)
   

def arrange_data(data_dict: dict):
    unique_labels = unique_data_labels(data_dict)
    data_dims = [data_dict[sheet]['size'][0] for sheet in data_dict.keys()]
    
    if len(set(data_dims)) != 1:
        raise Exception("inconsistent data dimensions")
    else:
        label_data_dict = {}
        for label in unique_labels:
            label_data = np.array([], dtype = np.float64).reshape(data_dims[0], 0)
            for sheet in data_dict.keys():
                data_index = [l == label for l in data_dict[sheet]['labels']]
                np.hstack([label_data, data_dict[sheet]['values'][:, data_index]])
            
            label_data_dict[label] = input_clean.clean_data(label_data)

    return label_data_dict

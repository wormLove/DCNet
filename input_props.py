import pandas as pd
import numpy as np
from typing import List
import warnings

def set_data_range(data: np.ndarray):
    min_zero_data = data - np.min(data, axis = 0)
    data_p = min_zero_data / np.max(min_zero_data, axis = 0)
    
    return data_p

def remove_zero_cols(data: np.ndarray):
    good_index = np.sum(data, axis = 0) != 0
    data_p = data[:, good_index]
    
    return data_p

def remove_nans(data: np.ndarray):
    data[np.isnan(data)] = 0

def clean_data(label_data):
    remove_nans(label_data)
    data_p = remove_zero_cols(label_data)
    data_p = set_data_range(data_p)
    remove_nans(data_p)
    
    return data_p

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
            
            label_data_dict[label] = clean_data(label_data)

    return label_data_dict

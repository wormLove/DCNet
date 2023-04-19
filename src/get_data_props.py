import pandas as pd
import numpy as np
from typing import List
import warnings

def get_unique_data_labels(data_dict:dict):
    list_of_lists = [data_dict[sheet]['labels'] for sheet in data_dict.keys()]
    list_of_labels = []
    for each_list in list_of_lists:
        for each_label in each_list:
            list_of_labels.append(each_label)

    unique_labels = set(list_of_labels)
    return unique_labels

def get_nsamples(data_dict:dict):
    nsamples = 0
    for sheet in data_dict.keys():
        nsamples += data_dict[sheet]['size'][1]

    return nsamples

def get_nclass(data_dict:dict):
    unique_labels = get_unique_data_labels(data_dict)
    return len(unique_labels)
   

def organize_data_with_labels(data_dict:dict):
    unique_labels, nclass = get_unique_data_labels(data_dict)
    data_dims = [data_dict[sheet]['size'][0] for sheet in data_dict.keys()]
    
    if len(set(data_dims)) != 1:
        raise Exception("inconsistent data dimensions")
    else:
        organized_data = {}
        
        for label in unique_labels:
            labeled_data = np.array([], dtype = np.float64).reshape(data_dims[0], 0)
            for sheet in data_dict.keys():
                data_index = [l == label for l in data_dict[sheet]['labels']]
                np.hstack([labeled_data, data_dict[sheet]['values'][:, data_index]])
            
            organized_data[label] = labeled_data

    return organized_data
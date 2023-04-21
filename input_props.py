import pandas as pd
import numpy as np
from typing import List
import warnings

import input_clean

def max_sheet_props(data_dict: dict):
    max_sheet_dim = 0
    for sheet in data_dict.keys():
        sheet_dim = data_dict[sheet]['size'][0]
        if sheet_dim > max_sheet_dim:
            max_sheet_dim = sheet_dim

    return max_sheet_dim

def conform_data(data_dict: dict):
    max_dim = max_sheet_props(data_dict)
    for sheet in data_dict.keys():
        sheet_dim = data_dict[sheet]['size'][0]
        pad = np.zeros((max_dim - sheet_dim, data_dict[sheet]['size'][1]))
        
        data_dict[sheet]['values'] = np.vstack((data_dict[sheet]['values'], pad))

    return max_dim

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
    data_dim = conform_data(data_dict)
    
    labeled_data_dict = {}
    for label in unique_labels:
        label_data = np.array([], dtype = np.float64).reshape(data_dim, 0)
        for sheet in data_dict.keys():
            data_index = [l == label for l in data_dict[sheet]['labels']]
            np.hstack([label_data, data_dict[sheet]['values'][:, data_index]])
            
            labeled_data_dict[label] = input_clean.clean_data(label_data)

    return labeled_data_dict

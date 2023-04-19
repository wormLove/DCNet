import numpy as np

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
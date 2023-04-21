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

def clean_data(labeled_data_dict: dict):
    for label, data in labeled_data_dict.items():
        remove_nans(data)
        labeled_data_dict[label] = remove_zero_cols(data)
        labeled_data_dict[label] = set_data_range(labeled_data_dict[label])
        remove_nans(labeled_data_dict[label])

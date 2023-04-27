import numpy as np

def remove_empty_keys(labeled_data_dict: dict):
    '''Function to remove labels without any data'''
    
    # create an empty dictionary to store good labels
    labeled_data_dict_clean = {}
    
    # loop through all labels and corresponding data
    for label, data in labeled_data_dict.items():
        
        # if data shape is 0, dont copy label data pair
        if data.shape[1] == 0:
            continue
        else:
            labeled_data_dict_clean[label] = data
    
    # return clean dictionary
    return labeled_data_dict_clean

def set_data_range(data: np.ndarray):
    '''Function to set data range between 0 and 1'''
    min_zero_data = data - np.min(data, axis = 0)
    data_p = min_zero_data / np.max(min_zero_data, axis = 0)
    return data_p

def remove_zero_cols(data: np.ndarray):
    '''Function to remove all columns with all zeros'''
    good_index = np.sum(data, axis = 0) != 0
    data_p = data[:, good_index]
    return data_p

def remove_nans(data: np.ndarray):
    '''Function to remove nans in-place'''
    data[np.isnan(data)] = 0

def clean_data(labeled_data_dict: dict):
    '''Function to clean data in a dictionary'''
    
    # loop through data for each label and clean the data
    for label, data in labeled_data_dict.items():
        remove_nans(data)
        labeled_data_dict[label] = remove_zero_cols(data)
        labeled_data_dict[label] = set_data_range(labeled_data_dict[label])
        remove_nans(labeled_data_dict[label])
    
    # remove any labels that do not hold any data and return clean dictionary
    labeled_data_dict_clean = remove_empty_keys(labeled_data_dict)
    return labeled_data_dict_clean

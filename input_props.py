import numpy as np
from typing import List
import math

import input_clean, input_save

def training_data_props(train_data_dict: dict):
    '''Function to get class names, and minimum class size from training data dictionary'''
    
    # initialize property values
    min_number_of_samples = 0
    
    # names of classes are the keys in the training data dictionary
    class_names = list(train_data_dict.keys())
    
    # loop through all classes in the training data dictionary and update property values
    for label in train_data_dict:
        data = train_data_dict[label]
        data_size = data.shape[1]
        
        # check for minimum class size
        if min_number_of_samples < data_size:
            min_number_of_samples = data_size
    
    # return property values
    return class_names, min_number_of_samples


def partition_dict(labeled_data_dict: dict, fraction: float):
    '''Function to partition the data in training and test sets (called only when test sheets are absent)'''
    
    # loop through all data labels and partition data into training and test groups (80% and 20% respectively)
    train_data_dict = {}
    test_data_dict = {}
    for label, data in labeled_data_dict.items():
        partition_index = math.ceil(fraction*data.shape[1])
        train_data_dict[label] = data[:, :partition_index]
        test_data_dict[label] = data[:, partition_index:]
    
    # return train and test data dictionaries 
    return train_data_dict, test_data_dict


def get_labeled_data(data_dict: dict, unique_labels: set, data_dim: int, sheets:List[str]):
    '''Fuction to organiz the data according to data labels'''
    
    # loop through all uniqe labels in the data
    labeled_data_dict = {}
    for label in unique_labels:
        label_data = np.array([], dtype = np.float64).reshape(data_dim, 0)
        
        # loop throgh all sheets in the data file
        for sheet in sheets:
            data_index = [l == label for l in data_dict[sheet]['labels']]
            sliced_data = data_dict[sheet]['values'][:, data_index]
            label_data = np.hstack((label_data, sliced_data))
        
        # put label data pairs into a dictionary
        labeled_data_dict[label] = label_data
    
    # retrn the dictionary 
    return labeled_data_dict


def get_data_dicts(data_dict: dict, unique_labels: set, data_dim: int, test_sheets: List[str]):
    '''Function to obtain train and test data dictionaries'''

    # if there are test sheets, organize the data into test and train dictionaries
    if test_sheets:
        train_sheets = [sheet for sheet in data_dict.keys() if sheet not in test_sheets]
        train_data_dict = get_labeled_data(data_dict, unique_labels, data_dim, train_sheets)
        test_data_dict = get_labeled_data(data_dict, unique_labels, data_dim, test_sheets)
    
    # otherwise divide the data (80% training and 20% test) and organize them into respective ditionaries
    else:
        all_sheets = [sheet for sheet in data_dict.keys()]
        labeled_data_dict = get_labeled_data(data_dict, unique_labels, data_dim, all_sheets)
        train_data_dict, test_data_dict = partition_dict(labeled_data_dict, 0.8)
    
    # return the training and the test dictionaries
    return train_data_dict, test_data_dict


def get_test_sheets(data_dict: dict):
    '''Function to get names of any test sheet (must start with the keyword "test")'''
    
    # Loop through all sheet names to get a list of all test sheet names
    test_sheets = []
    for sheet in data_dict.keys():
        sheet_lower = sheet.lower()
        if sheet_lower.startswith('test'):
            test_sheets.append(sheet)
    
    # return test sheet names
    return test_sheets


def max_sheet_props(data_dict: dict):
    '''Function to get the size of the largest data vector across all sheet data'''
    
    # loop through all sheets to get the largest dimension
    max_sheet_dim = 0
    for sheet in data_dict.keys():
        sheet_dim = data_dict[sheet]['size'][0] 
        if sheet_dim > max_sheet_dim:
            max_sheet_dim = sheet_dim
    
    # return the maximum dimension
    return max_sheet_dim


def conform_data(data_dict: dict):
    '''Function to make data vectors from all sheets consistent in size by padding'''
    
    # get the largest dimension of the data vectors from all sheets
    max_dim = max_sheet_props(data_dict)
    
    # pad the vectors to make them consisent in size
    for sheet in data_dict.keys():
        sheet_dim = data_dict[sheet]['size'][0]
        pad = np.zeros((max_dim - sheet_dim, data_dict[sheet]['size'][1]))
        
        data_dict[sheet]['values'] = np.vstack((data_dict[sheet]['values'], pad))
    
    # return the size of the largest vector
    return max_dim


def unique_data_labels(data_dict: dict):
    '''Function to get unique class names'''
    
    # create a list of all class labels
    list_of_lists = [data_dict[sheet]['labels'] for sheet in data_dict.keys()]
    list_of_labels = []
    for each_list in list_of_lists:
        for each_label in each_list:
            list_of_labels.append(each_label)
    
    # get unique class labels using set
    unique_labels = set(list_of_labels)
    
    # return unique labels
    return unique_labels


def organize_data(data_dict: dict):
    '''Function to organize data read from the data file'''
    
    # get unique data labels, dimension of the data vectors, and any test sheets present
    unique_labels = unique_data_labels(data_dict)
    data_dim = conform_data(data_dict)
    test_sheets = get_test_sheets(data_dict)
    
    #get data dictionaries and clean data in them
    train_data_dict, test_data_dict = get_data_dicts(data_dict, unique_labels, data_dim, test_sheets)
    input_clean.clean_data(train_data_dict)
    input_clean.clean_data(test_data_dict)
    
    # get number of samples, number_of_classes, and size of each class in the training dictionary
    class_names, min_number_of_samples = training_data_props(train_data_dict)
    
    # save collected values in db file
    input_save.save_organized_data(class_names, data_dim, min_number_of_samples, train_data_dict, test_data_dict)



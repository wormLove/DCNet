import pandas as pd
import numpy as np
from typing import List
import math

import input_clean

def nsamples(data_dict: dict):
    nsamples = 0
    for sheet in data_dict.keys():
        nsamples += data_dict[sheet]['size'][1]
    return nsamples


def nclass(data_dict: dict):
    unique_labels = unique_data_labels(data_dict)
    return len(unique_labels)

def partition_dict(labeled_data_dict: dict, fraction: float):
    train_data_dict = {}
    test_data_dict = {}
    for label, data in labeled_data_dict.items():
        partition_index = math.ceil(fraction*data.shape[1])

        train_data_dict[label] = data[:, :partition_index]
        test_data_dict[label] = data[:, partition_index:]
    return train_data_dict, test_data_dict


def get_labeled_data(data_dict: dict, unique_labels: set, data_dim: int, sheets:List[str]):
    labeled_data_dict = {}
    for label in unique_labels:
        label_data = np.array([], dtype = np.float64).reshape(data_dim, 0)
        for sheet in sheets:
            data_index = [l == label for l in data_dict[sheet]['labels']]
            sliced_data = data_dict[sheet]['values'][:, data_index]
            label_data = np.hstack((label_data, sliced_data))
        labeled_data_dict[label] = label_data
    return labeled_data_dict


def get_data_dicts(data_dict: dict, unique_labels: set, data_dim: int, test_sheets: List[str]):
    if test_sheets:
        train_sheets = [sheet for sheet in data_dict.keys() if sheet not in test_sheets]
        train_data_dict = get_labeled_data(data_dict, unique_labels, data_dim, train_sheets)
        test_data_dict = get_labeled_data(data_dict, unique_labels, data_dim, test_sheets)
    else:
        all_sheets = [sheet for sheet in data_dict.keys()]
        labeled_data_dict = get_labeled_data(data_dict, unique_labels, data_dim, all_sheets)
        train_data_dict, test_data_dict = partition_dict(labeled_data_dict, 0.8)
    return train_data_dict, test_data_dict


def get_test_sheets(data_dict: dict):
    test_sheets = []
    for sheet in data_dict.keys():
        sheet_lower = sheet.lower()
        if sheet_lower.startswith('test'):
            test_sheets.append(sheet)
    return test_sheets


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


def organize_data(data_dict: dict):
    unique_labels = unique_data_labels(data_dict)
    data_dim = conform_data(data_dict)
    test_sheets = get_test_sheets(data_dict)

    train_data_dict, test_data_dict = get_data_dicts(data_dict, unique_labels, data_dim, test_sheets)
    input_clean.clean_data(train_data_dict)
    input_clean.clean_data(test_data_dict)

    number_of_samples = nsamples(data_dict)
    number_of_classes = nclass(data_dict)

    return number_of_classes, number_of_samples, train_data_dict, test_data_dict


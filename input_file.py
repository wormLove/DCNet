import pandas as pd
import numpy as np
from typing import List
import warnings

def process_labels(data_labels: List[str]):
    '''Function to process duplicate labels'''
    processed_labels = [s.split('.')[0] for s in data_labels]
    return processed_labels


def get_sheet_data(f: dict, sheet: str):
    '''Function to get sheet data as array and its size'''
    df = f[sheet]
    data_as_array = df.to_numpy()
    data_size = data_as_array.shape
    return data_as_array, data_size

def get_sheet_labels(f: dict, sheet: str, ncols: int, header: bool):
    '''Function to get data labels from the sheet'''
    df = f[sheet]
    
    # if data labels or sheed headers are present
    if header:
        data_labels_raw = [str(value) for value in list(df.columns.values)]
        data_labels = process_labels(data_labels_raw)
        
        # check if number of labels match the number of data points
        if len(data_labels) != ncols:
            raise Exception("improper labels")
    
    # otherwise set all data labels to '<ukn>'
    else:
        data_labels = ['<ukn>']*ncols
    
    # return data labels
    return data_labels

def read_sheet(f: dict, sheet: str, header: bool = True):
    '''Function to read sheet data i.e. data labels, data array, and data size'''
    
    # create a dictionary to store data from each sheet
    sheet_dict = {}
    
    # get sheet data and store it in the dictionary
    data_values, data_size = get_sheet_data(f, sheet)
    sheet_dict['values'] = data_values
    sheet_dict['size'] = data_size
    sheet_dict['labels'] = get_sheet_labels(f, sheet, int(data_size[1]), header)
    
    # return the sheet dictionary
    return sheet_dict

def read_data(file_path: str, header: bool = True, sheets: List[str] = []):
    '''Function to read data from an excel/csv file'''
    
    # read file
    f = pd.read_excel(file_path, None)
    
    # extract all sheet names as list of strings
    sheet_names_from_file = [str(key) for key in f.keys()]
    
    # organize data into a dictionary with keys as sheet names and values as other dictionary conatining labesl, data array, and data size 
    data_dict = {}
    
    # if there are user defined sheet to read, read sheets that actually exists, otherwise throw warning
    if sheets:
        for sheet in sheets:
            if sheet in sheet_names_from_file:
                data_dict[sheet] = read_sheet(f, sheet, header=header)          
            else:
                warnings.warn(f"sheet {sheet} not in file")
    
    # else read all sheets
    else:
        for sheet in sheet_names_from_file:
            data_dict[sheet] = read_sheet(f, sheet)
    
    # return data dictionary
    return data_dict

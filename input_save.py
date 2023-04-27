import shelve
import os
from datetime import datetime
from typing import List

def date_stamp():
    '''Function to generate a date stamp'''
    
    # generate current date stamp
    now = datetime.now()
    date_stamp = now.strftime("-%b%d-%H%M")
    
    #return date stamp
    return date_stamp

def save_organized_data(class_names: List[str], data_dim: int, min_number_of_samples: int, train_data_dict: dict, test_data_dict: dict):
    '''Function to save the organized data'''
    
    # generate path to data directory, create if doesn't esist
    data_dir = os.path.join(os.getcwd(), "Data")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    # cerate a file name and file path
    fid = date_stamp()
    filename = "Data" + fid
    file_path = os.path.join(data_dir, filename)
    
    # create a data file with created filename at filepath
    f = shelve.open(file_path)
    f['class_names'] = class_names
    f['data_dim'] = data_dim
    f['max_batch_size'] = min_number_of_samples*len(class_names)
    f['train_dict'] = train_data_dict
    f['test_dict'] = test_data_dict
    f.close()
    
    # return fid
    return fid
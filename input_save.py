import shelve
import os
from datetime import datetime

def date_stamp():
    '''Function to generate a date stamp'''
    
    # generate current date stamp
    now = datetime.now()
    date_stamp = now.strftime("-%b%d-%H%M")
    
    #return date stamp
    return date_stamp

def save_organized_data(number_of_classes: int, number_of_samples: int, unique_labels: set, train_data_dict: dict, test_data_dict: dict):
    '''Function to save the organized data'''
    
    # generate path to data directory, create if doesn't esist
    data_dir = os.path.join(os.getcwd(), "Data")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    # cerate a file name and file path
    filename = "Data" + date_stamp()
    file_path = os.path.join(data_dir, filename)
    
    # create a data file with created filename at filepath
    f = shelve.open(file_path)
    f['nclass'] = number_of_classes
    f['nsamples'] = number_of_samples
    f['labels'] = unique_labels
    f['train_dict'] = train_data_dict
    f['test_dict'] = test_data_dict
    f.close()
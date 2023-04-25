import numpy as np
from typing import List
import warnings
import random
import shelve
import os

import input_props, input_file

class Data:
    '''Define parent class Data'''
    
    # define constructor for class 
    def __init__(self, fid: str):
        # initialize class variables with None
        self.nclass = None
        self.nsamples = None
        self.labels = None
        
        # call function to load data
        self.load_data(fid)
    
    # define a function to load data into object
    def load_data(self, fid: str, data_type: str = ""):
        # get path to db data file
        path_to_file = self.process_fid(fid)
        
        # load data from file for different data types
        f = shelve.open(path_to_file, 'r')
        if not data_type:
            self.nclass = f['nclass']
            self.nsamples = f['nsamples']
            self.labels = f['labels']
        # check if train or test data need to be loaded
        elif data_type == 'train':
            self.data = f['train_dict']
        elif data_type == 'test':
            self.data = f['test_dict']
        
        # close file
        f.close()

    
    # define a function check if fid is valid and return valid path to data file
    def process_fid(self, fid: str):
        # create path to db file using fid
        data_dir = os.path.join(os.getcwd(), "Data")
        filename = "Data-" + fid
        file_path = os.path.join(data_dir, filename)
        
        # if path to file is present return it, else raise exception
        if os.path.isfile(file_path):
            return file_path
        else:
            raise Exception("file with fid not found in current directory") 


class Train(Data):
    '''Create a class for training data'''
    
    # define constructor for Train object 
    def __init__(self, fid: str):
        # initialize all object variables from the parent class and local data variable
        super().__init__(fid)
        self.data = None
        
        # call parent load_data function to load data
        self.load_data(fid, data_type = 'train')
    
    # define the next function for Train object
    def __next__(self):
        pass


class Test(Data):
    '''Create class for test data'''
    
    # define constructor for Test object
    def __init__(self, fid: str):
        # initialize local data variable
        self.data = None
        
        # call parent load_data function to load data
        self.load_data(fid, data_type = 'test')
    
    # define next function for test object
    def __next__(self):
        pass



def get_input_object(file_path: str, header: bool = True, sheets: List[str] = []):
    '''Function to create input object'''
    
    # get data dictionary from the csv/excel file
    data_dict = input_file.read_data(file_path, header = header, sheets = sheets)
    
    # organize and save the data into a db file
    input_props.organize_data(data_dict)
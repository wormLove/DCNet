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
        # initialize class variables
        self.class_names = []
        self.data_dim = 0
        
        # call function to load data
        self.load_data(fid)
    
    # define a function to load data into object
    def load_data(self, fid: str, data_type: str = ""):
        # get path to db data file
        path_to_file = self.process_fid(fid)
        
        # load data from file for different data types
        f = shelve.open(path_to_file, 'r')
        if not data_type:
            self.class_names = f['class_names']
            self.data_dim = f['data_dim']
        
        # check if train data need to be loaded
        elif data_type == 'train':
            self.data = f['train_dict']
            self.max_batch_size = f['max_batch_size']
        
        # check if test data need to be loaded
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
        if os.path.isfile(file_path + ".db"):
            return file_path
        else:
            raise Exception("file with fid not found in current directory") 


class Train(Data):
    '''Create a class for training data'''
    
    # define constructor for Train object 
    def __init__(self, fid: str):
        # initialize all object variables from the parent class and local data variable
        super().__init__(fid)
        self.data = {}
        self.max_batch_size = 0
        self.training_batch = np.array([], dtype = np.float64).reshape(self.data_dim, 0)
        self.train_sample_return_index = 0
        
        # call parent load_data function to load data
        self.load_data(fid, data_type = 'train')
    
    # define iter function
    def __iter__(self):
        return self
    
    # define the next function
    def __next__(self):
        if self.train_sample_return_index < self.training_batch.shape[1]:
            self.train_sample_return_index += 1
            return self.training_batch[:, self.train_sample_return_index-1].reshape(self.data_dim, 1)
        else:
            raise StopIteration   
    
    # define a function to adjust batch size according to data
    def get_class_samples_in_batch(self, number_of_classes: int, batch_size: int):
        # if batch size is larger than maximum possible batch size adjust batch size to maximum possible batch size 
        if batch_size > self.max_batch_size:
            class_samples_in_batch = self.max_batch_size//number_of_classes
            warnings.warn(f"batch size too large, adjusted to {self.max_batch_size}")
        
        # else adjust it to closest possible value where all classes are represented equally
        else:
            class_samples_in_batch = batch_size//number_of_classes
            adjusted_batch_size = class_samples_in_batch*number_of_classes
            if adjusted_batch_size != batch_size:
                warnings.warn(f"batch size adjusted to {adjusted_batch_size}")
        
        # return number of samples from each class in the batch
        return class_samples_in_batch
    
    # define a function to form training batch and initialize training procedure
    def initialize_batch(self, batch_size: int):
        # get number of classes and initialize a training batch array
        number_of_classes = len(self.class_names)
        
        # calculate the number of samples from each class to be in the batch
        class_samples_in_batch = self.get_class_samples_in_batch(number_of_classes, batch_size)
        
        # create a list of class indices and shuffle it
        class_index_list = list(range(number_of_classes))
        random.shuffle(class_index_list)
        
        # loop through all class indices
        for class_index in class_index_list:
            # select the class label corresponding to the class index and obtain its data
            label = self.class_names[class_index_list[class_index]]
            label_data = self.data[label]
            
            # choose a random start point in the data and stack slice of data
            sample_index_start = random.randint(0, label_data.shape[1] - class_samples_in_batch)
            label_data_to_stack = label_data[:, sample_index_start:sample_index_start+class_samples_in_batch]
            self.training_batch = np.hstack((self.training_batch, label_data_to_stack))
            
            # reset the train sample return index to 0
            self.train_sample_return_index = 0

class Test(Data):
    '''Create class for test data'''
    
    # define constructor for Test object
    def __init__(self, fid: str):
        # initialize local data variable
        super().__init__(fid)
        self.data = {}
        self.test_batch = np.array([], dtype = np.float64).reshape(self.data_dim, 0)
        self.test_sample_return_index = 0
        
        # call parent load_data function to load data
        self.load_data(fid, data_type = 'test')
    
    # define iter function
    def __iter__(self):
        return self
    
    # define next function for test object
    def __next__(self):
        if  self.test_sample_return_index < self.test_batch.shape[1]:
            self.test_sample_return_index += 1
            return self.test_batch[:,  self.test_sample_return_index-1]
        else:
            raise StopIteration 
    
    # define a function to form test batch and initialize test procedure
    def initialize_batch(self):
        for _, data in self.data.items():
            self.test_batch = np.hstack((self.test_batch, data))

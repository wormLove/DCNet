import numpy as np
from typing import List
import warnings
import random

import input_props, input_file

class InputObj:
    def __init__(self, nclass, nsamples, labeled_data_dict):
        self.nclass = nclass
        self.nsamples = nsamples
        self.data = labeled_data_dict


    def get_train_data(self, label, *n_inps):
        self.check_args(label, *n_inps)
        data_size = self.get_data_size(label)
        if not n_inps:
            data_index = random.randint(0, data_size)
            return self.data[label][:, data_index]
        else:
            random_start = random.randint(0, data_size - n_inps[0])
            return self.data[label][:, random_start:random_start+10]

    def get_test_data(self, label, *n_inps):
        self.check_args(label, *n_inps)

    def get_data_size(self, label):
        return self.data[label].shape[1]
    
    def check_args(self, label, *n_inps):
        data_size = self.get_data_size(label)
        if n_inps:
            if len(n_inps) > 1:
                raise Exception("more than one number of inputs given")
            else:
                if n_inps[0] > data_size:
                    raise Exception("number of inputs exceets data size")
        if label not in self.data:
            raise Exception(f"label {label} not found")
        
    def label_list(self):
        return [key for key in self.data.keys()]


def get_input_object(file_path: str, header: bool = True, sheets: List[str] = []):
    data_dict = input_file.read_data(file_path, header = header, sheets = sheets)
    
    nclass = input_props.nclass(data_dict)
    nsamples = input_props.nsamples(data_dict)
    labeled_data_dict = input_props.arrange_data(data_dict)
    
    return InputObj(nclass, nsamples, labeled_data_dict)
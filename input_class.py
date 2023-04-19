import numpy as np
from typing import List
import warnings

import input_props, input_file

class InputObj:
    def __init__(self, nclass, nsamples, train_set, test_set):
        self.nclass = nclass
        self.nsamples = nsamples
        self.train = train_set
        self.test = test_set


def get_train_data(label_data_dict: dict):
    pass

def get_test_data(label_data_dict: dict):
    pass

def get_input_object(file_path: str, header: bool = True, sheets: List[str] = []):
    data_dict = input_file.read_data(file_path, header = header, sheets = sheets)
    
    nclass = input_props.nclass(data_dict)
    nsamples = input_props.nsamples(data_dict)
    label_data_dict = input_props.arrange_data(data_dict)
    
    train_set = get_train_data(label_data_dict)
    test_set = get_test_data(label_data_dict)
    
    return InputObj(nclass, nsamples, train_set, test_set)
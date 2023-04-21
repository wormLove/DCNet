import numpy as np
from typing import List
import warnings
import random

import input_props, input_file

class InputObj:
    def __init__(self, nclass, nsamples, train_data_dict, test_data_dict):
        self._nclass = nclass
        self._nsamples = nsamples
        self._train_data = train_data_dict
        self._test_data = test_data_dict


def get_input_object(file_path: str, header: bool = True, sheets: List[str] = []):
    data_dict = input_file.read_data(file_path, header = header, sheets = sheets)
    number_of_classes, number_of_samples, train_data_dict, test_data_dict = input_props.organize_data(data_dict)
    
    return InputObj(number_of_classes, number_of_samples, train_data_dict, test_data_dict)
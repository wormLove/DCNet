from typing import List
import shelve
import numpy as np

import input_props, input_file
from input_class import Train

def check_data_dict(data_dict: dict):
    for sheet in data_dict:
        print(sheet)
        print(data_dict[sheet]['size'])

def check_train_dict(class_names, data_dim, max_batch_size, train_dict, data_dict):
    print(f"number of classes == number of keys : {len(class_names) == len(train_dict.keys())}")
    print(f"keys == class names : {class_names.sort() == list(train_dict.keys()).sort()}")
    
    min_number_of_samples = np.inf
    shape_consistency = True
    labels_without_data = False
    for label, data in train_dict.items():
        if data.shape[1] < min_number_of_samples:
            min_number_of_samples = data.shape[1]

        if data.shape[0] != data_dim:
            shape_consistency = False

        if data.shape[1] == 0:
            labels_without_data = True
    
    max_sheet_dim = 0
    for sheet in data_dict.keys():
        if data_dict[sheet]['size'][0] > max_sheet_dim:
            max_sheet_dim = data_dict[sheet]['size'][0]

    
    print(f"max batch size correct : {max_batch_size == min_number_of_samples*len(class_names)}")
    print(f"shape consistency : {shape_consistency}")
    print(f"labels without data : {labels_without_data}")
    print(f"data dim == max sheet dim : {data_dim == max_sheet_dim}")

def check_input_class(train_unputs, data_dict):
    print("**----------cheking input class----------**")
    check_train_dict(train_inputs.class_names, train_inputs.data_dim, train_inputs.max_batch_size, train_inputs.data, data_dict)
    
    adjusted_batch_size = train_inputs.training_batch.shape[1]
    number_of_classes = len(train_inputs.class_names)
    q, r = divmod(adjusted_batch_size, number_of_classes)
    print(f"proper batch adjustemet : {r == 0}")

    train_inputs_consistency = True
    for input in train_inputs:
        train_inputs_consistency = train_inputs_consistency and input.shape[0] == train_inputs.data_dim
        train_inputs_consistency = train_inputs_consistency and input.shape[1] == 1
    print(f"train input consistency : {train_inputs_consistency}")




def get_input_object(file_path: str, header: bool = True, sheets: List[str] = []):
    '''Function to create input object'''
    
    # get data dictionary from the csv/excel file
    data_dict = input_file.read_data(file_path, header = header, sheets = sheets)
    
    # organize and save the data into a db file
    input_props.organize_data(data_dict)
    return data_dict

data_dict = get_input_object('/Users/rraj/PythonFunctions/DCNet/GlomData_clean.xls')
check_data_dict(data_dict)

f = shelve.open('/Users/rraj/PythonFunctions/DCNet/Data/Data-Apr27-1516', 'r')
class_names = f['class_names']
data_dim = f['data_dim']
max_batch_size = f['max_batch_size']
train_dict = f['train_dict']
test_dict = f['test_dict']
f.close()

check_train_dict(class_names, data_dim, max_batch_size, train_dict, data_dict)

train_inputs = Train('Apr27-1516')
train_inputs.initialize_training_procedure(100)
check_input_class(train_inputs, data_dict)
import pandas as pd
import numpy as np
from typing import List
import warnings

def get_sheet_data(f: dict, sheet: str):
    df = f[sheet]

    data_as_array = df.to_numpy()
    data_size = data_as_array.shape
    
    return data_as_array, data_size

def get_sheet_labels(f: dict, sheet: str, ncols: int, header: bool):
    df = f[sheet]
    
    if header:
        data_labels = [str(value) for value in list(df.columns.values)]
        if len(data_labels) != ncols:
            raise Exception("improper labels")
    else:
        data_labels = ['<ukn>']*ncols

    return data_labels
        

def read_sheet(f: dict, sheet: str, header: bool = True):
    sheet_dict = {}
    
    data_values, data_size = get_sheet_data(f, sheet)
    sheet_dict['values'] = data_values
    sheet_dict['size'] = data_size
    sheet_dict['labels'] = get_sheet_labels(f, sheet, int(data_size[1]), header)

    return sheet_dict

def read_data(file_path: str, header: bool = True, sheets: List[str] = []):
    f = pd.read_excel(file_path, None)
    sheet_names_from_file = str(set(f.keys()))
    
    data_dict = {}
    
    if sheets:
        for sheet in sheets:
            if sheet in sheet_names_from_file:
                data_dict[sheet] = read_sheet(f, sheet, header=header)          
            else:
                warnings.warn(f"sheet {sheet} not in file")
    else:
        for sheet in sheet_names_from_file:
            data_dict[sheet] = read_sheet(f, sheet)
        
    return data_dict

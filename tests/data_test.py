# This file is used to test any important change in the project
import pandas as pd
import numpy as np
import sys
import os 
#module_dir = os.path.abspath("D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\src")
#sys.path.append(module_dir)
import src.data as data

if __name__ == '__main__':
    path_file = r"D:\Documentos\Vida_profesional\Coding\Projects\customer_churn_ml\data\raw\raw.csv"
    df = data.load_data(path_file)
    df = data.clean_basic_issues(df)
    print(df.head())
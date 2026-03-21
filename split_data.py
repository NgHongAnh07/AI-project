import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
dataset= pd.read_csv(r"CSVs/dataset.csv")

train_data,val_data = train_test_split(dataset,test_size=0.3)
train_data.to_csv(r"CSVs/train_df.csv",index=False)
val_data.to_csv(r"CSVs/val_df.csv",index=False)

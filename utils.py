
from dataset import StockDataset
from torch.utils.data import DataLoader
import numpy as np 
import os 
import pandas as pd

def time_series_data(dataset, l, n, p, scaler): 
    data_length = len(dataset) - l - n - p + 1
    print(data_length)
    data = dataset
    data = scaler.transform(data)
    x = []
    y = []
    for idx in range(data_length):
        idx2test = idx * 5 
        if idx2test + 5 >= data_length : break   # cắt đuôi

        x_idx = data[idx2test:idx2test+l].reshape(1,-1)
        # y_idx = data[idx+l+n: idx+l+n+p] + data[idx+l-1 : idx+l]
        y_idx = data[idx2test+l+n: idx2test+l+n+p] 

        
        x.append(x_idx)
        y.append(y_idx.transpose())
    x = np.array(x)
    y = np.array(y)
    return x, y 

def get_set_and_loader(data, hyperconfig, shuffle = True, scale_to_test=None):
    # Create dataset and loader from data frame
    dataset = StockDataset(data, hyperconfig, scale_to_test)

    loader = DataLoader(dataset = dataset, 
                        batch_size = hyperconfig["n_way"], 
                        shuffle = shuffle)

    return dataset, loader

def load_time_data(folder, src_name, time):
    path = os.path.join(folder, f"{src_name}.csv")
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)

    start_day = time["start"]
    finish_day = time["finish"]
    df = df.sort_values(by=df.columns[0])
    df = df[(df['time'] > time['start']) & (df['time'] < time['finish'])]
    array = df[df.columns[1]].to_numpy()

    # time_cut_df = df[(df['datetime'] > start_day) & (df['datetime'] < finish_day)]
    # data = time_cut_df.close.to_numpy()
    is_array_nan = pd.isnull(df[df.columns[1]].to_frame()).to_numpy().squeeze()
    flag = False
    start = 0
    data = []
    for i in range(len(array)):
        if flag == False:
            if is_array_nan[i] == False:
                start = i
                flag = True
        else:
            if is_array_nan[i] == True or i == array.shape[0] - 1:
                flag = False
                if i - start > 100 :  
                    data.append(np.expand_dims(array[start:i], axis=1))
    return data

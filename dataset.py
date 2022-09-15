from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

class StockDataset(Dataset):
    def __init__(self, corps, hyperconfig, n_way ,scale_to_test=None):
        self.l = hyperconfig["l"]
        self.n = hyperconfig["n"]
        self.p = hyperconfig["p"]
        
        self.k_shot = hyperconfig["k_shot"]
        self.k_query = hyperconfig["k_query"]
        self.n_way = n_way

        # Xử lý scale : 
        
        if scale_to_test is None : 
            self.scaler = MinMaxScaler()
            self.data = [self.scaler.fit_transform(corp) for corp in corps]
        else : 
            self.scaler = scale_to_test
            print(corp for corp in corps)
            self.data = [self.scaler.transform(corp) for corp in corps]

        
        episode_lengths = [(len(x) - self.l - self.n - self.p + 1) // (self.k_shot + self.k_query) for x in self.data]
        self.episodes = min(episode_lengths)
        
        self.create_batches()
        
        
    def reverse_normalize(self, x):
        x = x.reshape(-1,1).to('cpu')
        x = self.scaler.inverse_transform(x)

        x = torch.tensor(x).to(dtype=torch.float32)
        return x
    
    def create_batches(self):
        self.support_x = []
        self.support_y = []
        self.query_x = []
        self.query_y = []
        
        x_full = []
        y_full = []
        
        for data in self.data:
            x = []
            y = []
            
            data_length = len(data) - self.l - self.n - self.p + 1
            for idx in range(data_length):
                x_idx = data[idx:idx+self.l].reshape(1,-1)
                y_idx = np.transpose(data[idx+self.l+self.n:idx+self.l+self.n+self.p])#.squeeze()

                x.append(x_idx)
                y.append(y_idx)
            
            x = np.array(x)
            y = np.array(y)#.reshape(-1,1,self.p)
            # print(y.shape)
            
            x_full.append(x)
            y_full.append(y)
            
        
        idx = 0
        for _ in range(self.episodes):
            selected_tasks = np.random.choice(list(range(len(self.data))), self.n_way, False)  # no duplicate
            np.random.shuffle(selected_tasks)
            
            for t in selected_tasks:
                data_x = x_full[t]
                data_y = y_full[t]
                
                selected_idx = np.random.choice(list(range(len(data_x))), (self.k_shot+self.k_query), False)
                np.random.shuffle(selected_idx)
                
                support_x_batch = data_x[selected_idx[:self.k_shot]]
                support_y_batch = data_y[selected_idx[:self.k_shot]]

                query_x_batch = data_x[selected_idx[self.k_shot:]]
                query_y_batch = data_y[selected_idx[self.k_shot:]]
            
                self.support_x.append(support_x_batch)
                self.support_y.append(support_y_batch)
                self.query_x.append(query_x_batch)
                self.query_y.append(query_y_batch)
                
                x_full[t] = np.delete(data_x, selected_idx, axis=0)
                y_full[t] = np.delete(data_y, selected_idx, axis=0)
        

    def __getitem__(self, idx):
        x_shot = self.support_x[idx]
        y_shot = self.support_y[idx]
        x_query = self.query_x[idx]
        y_query = self.query_y[idx]
        return x_shot, y_shot, x_query, y_query
    

    def __len__(self):
        return self.episodes



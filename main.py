import json
import os
import pandas as pd
from utils import load_time_data, get_set_and_loader 
import torch
from model import MAML
import numpy as np
import tqdm

train = []
test = {}

# read data
config = json.load(open("config.json", "r", encoding="utf8"))
source_folder = config["source_folder"]
source_name = config["source"]
target_name = config["target"]

for name in source_name:
    source_time = {
    "start": config["start-train"],
    "finish": config["finish-train"]
    }
    data = load_time_data(source_folder, name, source_time)
    train = train + data 

for name in target_name:
    target_time = {
    "start": "2022-06-08 23:59:59",
    "finish": "2023-01-01 00:00:00"
    }
    data = load_time_data(source_folder, name, target_time)
    
    test[name] = data
    # test.append(np.expand_dims(data, axis=1))

fine_tune_data = {}
for name in target_name:
    test_time = {
    "start": config["start-test"],
    "finish": config["finish-test"]
    }
    data = load_time_data(source_folder, name, test_time)
    fine_tune_data[name] = data
    #fine_tune_data.append(np.expand_dims(data, axis=1))

# buil model 

hyperconfig = json.load(open("hyperpara.json", "r"))

model_config = [
    ('linear', [128, hyperconfig["l"]]),
    ('relu', [True]),
 
    ('linear', [64, 128]),
    ('relu', [True]),   
   
    ('linear', [16, 64]),
    ('relu', [True]),

    ('linear', [hyperconfig["p"], 16])
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
maml = MAML(model_config, hyperconfig).to(device)

tmp = filter(lambda x: x.requires_grad, maml.parameters())
num = sum(map(lambda x: np.prod(x.shape), tmp))

print(maml)
print('Total trainable tensors:', num)


# training 
# print(train)
training_set_and_loader = get_set_and_loader(train, hyperconfig , True)
for epoch in tqdm(range(1)):
    # fetch meta_batchsz num of episode each time
    train_set, train_load = training_set_and_loader
    step = 0
    for x_spt, y_spt, x_qry, y_qry in train_load:
        step += 1 
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        # print()
        # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape) 
        # print()
        # print(x_spt.shape, y_spt.shape)
        accs = maml(x_spt, y_spt, x_qry, y_qry)
    
import json
import pandas as pd
from utils import load_time_data, get_set_and_loader , time_series_data, time_series_data_to_test
from utils import indicator
import torch
from model import MAML
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
    # print(len(data))
    train = train + data 

for name in target_name:
    target_time = {
    "start": "2022-06-08 23:59:59",
    "finish": "2023-01-01 00:00:00"
    }
    data = load_time_data(source_folder, name, target_time)
    # print(len(data))
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
    # ('linear', [128, hyperconfig["l"]]),
    # ('relu', [True]),
    ('lstm', [1, 64, True]),

     ('lstm_decoder', [1, 64, hyperconfig["p"]])
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
maml = MAML(model_config, hyperconfig).to(device)

tmp = filter(lambda x: x.requires_grad, maml.parameters())
num = sum(map(lambda x: np.prod(x.shape), tmp))

print(maml)
print('Total trainable tensors:', num)

epochs = hyperconfig["epochs"]
# training 
# print(train)
print("trian:  ", len(train))
print(hyperconfig["n_way"])
training_set_and_loader = get_set_and_loader(train, hyperconfig, hyperconfig["n_way"] , True, None)
for epoch in tqdm(range(epochs)):
    # print(epoch)
    # fetch meta_batchsz num of episode each time
    train_set, train_load = training_set_and_loader
    step = 0
    for x_spt, y_spt, x_qry, y_qry in train_load:
        step += 1 
        x_spt, y_spt=torch.permute(x_spt, (0,1,3,2)),torch.permute(y_spt, (0,1,3,2)) 
        x_qry, y_qry=torch.permute(x_qry, (0,1,3,2)),torch.permute(y_qry, (0,1,3,2)) 
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        # print()
        # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape) 
        # print()
        # print(x_spt.shape, y_spt.shape)

        accs = maml(x_spt, y_spt, x_qry, y_qry)
# print("train        ",x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape) 
metrics = {}

for name in target_name :
    # Lấy scaler 
    full =  np.array(np.concatenate((fine_tune_data[name] + test[name])))
    scaler = MinMaxScaler()
    scaler.fit_transform(full)

    tgt_data = fine_tune_data[name]
    test_set_and_loader = get_set_and_loader(tgt_data, hyperconfig, 1, False, scaler)
    test_set, test_load = test_set_and_loader
    for x_spt, y_spt, x_qry, y_qry in test_load:
        # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)

        x_spt, y_spt=torch.permute(x_spt, (0,1,3,2)),torch.permute(y_spt, (0,1,3,2)) 
        x_qry, y_qry=torch.permute(x_qry, (0,1,3,2)),torch.permute(y_qry, (0,1,3,2)) 
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
        # print("test     " , x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape) 
        accs, fast_weights = maml.finetunning(x_spt, y_spt, x_qry, y_qry)

   
    model = maml.net

    x , y = time_series_data(test[name][0], hyperconfig ,test_set.scaler)
    for i in range(len(test[name])-1) : 
        x_ , y_ = time_series_data(test[name][i+1], hyperconfig ,test_set.scaler)
        x , y = np.concatenate((x, x_)), np.concatenate((y, y_))

    x_tensor, y_tensor = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    x_tensor = torch.permute(x_tensor, (0,2,1)) #, torch.permute(y_tensor, (0,2,1))
    
    pred = model(x_tensor, vars=fast_weights, bn_training=False)
    
    # print(y_tensor.shape, pred.shape)
    label = test_set.reverse_normalize(y_tensor).numpy()
    pred1 = test_set.reverse_normalize(pred.detach()).numpy()
    metrics[name] = indicator(torch.tensor(pred1), torch.tensor(label))
    
    #plot 
    x , y = time_series_data_to_test(test[name][-1], hyperconfig ,test_set.scaler)
    x_tensor, y_tensor = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    x_tensor = torch.permute(x_tensor, (0,2,1))
    pred = model(x_tensor, vars=fast_weights, bn_training=False)
    
    label = test_set.reverse_normalize(y_tensor).numpy()
    pred1 = test_set.reverse_normalize(pred.detach()).numpy()
    fig = plt.figure() 
    fig.set_size_inches(40, 15)
    plt.plot(range(len(label)),label, label='Ground truth')
    plt.plot(range(len(pred1)), pred1, label='maml_mlp')
    plt.xlabel(name)
    plt.ylabel('Close')
    name_file = 'results/plot/maml_EnDecoder'+str(hyperconfig["p"])+name+'.png'
    fig.savefig(name_file)
    plt.legend()
    plt.show()


results = pd.DataFrame.from_dict(metrics, orient='index')
name_file_csv =  "results/metric_" + str(hyperconfig["p"]) + "_decoder_maml.csv"
results.to_csv(name_file_csv)



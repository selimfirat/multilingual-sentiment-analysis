import argparse
import numpy as np
import torch

from experiment_bc import ExperimentBinaryClassification

from pathlib import Path
import os
import copy
import json

np.random.seed(1)
torch.manual_seed(0)

def write_model(ds_name, model_str, model, hid):
    Path("hyper_models/" + str(hid)).mkdir(parents=True, exist_ok=True)
    torch.save(model, os.path.join("hyper_models/" + str(hid), ds_name + "_" + model_str + "_best.pt"))

parser = argparse.ArgumentParser(description='Hyperparam Selector for Sentiment Deep Metric Learning')
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--experiment", type=str, default="default")

cfg = vars(parser.parse_args())

#data = ['SE0714', 'Olympic', 'SS-Twitter', 'SS-Youtube', 'SCv1', 'SCv2-GEN']
#data = ['SE0714', 'Olympic', 'SS-Twitter', 'SS-Youtube']
data = ['SS-Youtube']
cfg['all_datasets'] = ['SS-Twitter', 'SS-Youtube', 'SCv1', 'SCv2-GEN', 'SE0714', 'Olympic', 'kaggle-insults']
cfg['ml_datasets'] = ['SE0714', 'Olympic', 'emoji-tweets']

models = ['rnn', 'lstm', 'gru']
Path("hyper_models").mkdir(parents=True, exist_ok=True)

params = {
    'num_epochs': 100, 'lr': 0.01, 'dropout': 0.5,
    'rec_dropout': 0, 'weight_decay': 1e-6, 'batch_size': 64,
    'patience': 10, 'pretrained': "bert-large-uncased", 'pooling': "mean",
    'hidden_size': 128, 'num_layers': 1, 'bidirectional': True,
    'fix_prior': True, 'metric_dim': 32, 'finetune': None, 
    'finetune_type': "change_final", 'model': "gru"
    }
params['hid'] = 0
params['device'] = cfg['device']
params['experiment'] = cfg['experiment']

#hidden_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
#num_layers = [1, 2, 4]
#batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256]
#dropouts = [0.1, 0.3, 0.5, 0.7, 0.9]
poolings = ['max']
exp_results = []

results = {}
hypers = []
for idx, ele in enumerate(poolings):
    param = copy.deepcopy(params)
    param['hid'] = ele
    param['poolings'] = ele
    hypers.append(param)
    results[ele] = {}

table_str = ""
for hyp in hypers:
    table_str = str(hyp['hid'])
    for datum in data:
        hyp['data'] = datum
        best_val = 100000
        results[hyp['hid']][datum] = {'str': -1, 'model': -1, 'val': -1, 'f1': -1, 'all': []}
        
        for model_str in models:
            print(f"Experimenting with the data {datum} using model {model_str}")
            hyp['model'] = model_str
            experiment = ExperimentBinaryClassification(hyp, True)
            val_loss, model, f1 = experiment.run()
            results[hyp['hid']][datum]['all'].append([model_str, f1, val_loss])
            table_str = table_str + " & " + str(f1)
            
            if val_loss < best_val:
                best_val = val_loss
                results[hyp['hid']][datum]['str'] = model_str
                results[hyp['hid']][datum]['val'] = best_val
                results[hyp['hid']][datum]['model'] = model
                results[hyp['hid']][datum]['f1'] = f1
                
        write_model(datum, results[hyp['hid']][datum]['str'], results[hyp['hid']][datum]['model'], hyp['hid'])
        results[hyp['hid']][datum]['model'] = ""
        exp_results.append([hyp['hid'], results[hyp['hid']][datum]['f1'], results[hyp['hid']][datum]['val'], results[hyp['hid']][datum]['str'], table_str])
        json.dump(results, open("hyper_models/" + str(hyp['hid']) + "/results.txt",'w'))
    
    
print(exp_results)
print(results)

import pandas as pd
import requests
import os
import json
import numpy as np
from tqdm import tqdm
import multiprocessing
import os
import gc

# cofig
feature_names = [
    'TransactionAmt',
    'ProductCD',
    'card1',
    'C1',
    'C2',
    'C3',
    'C4',
    'C5',
    'C6',
    'C7',
    'C8',
    'C9',
    'C10',
    'C11',
    'C12',
    'C13',
    'C14']
productCD_categories = ['C', 'H', 'R', 'S', 'W']
headers = {'Content-type': 'application/json'}
num_cpu = multiprocessing.cpu_count()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# load data
package_dir = os.path.dirname(os.path.abspath(__file__))
sample_submission = pd.read_csv(os.path.join(
    package_dir, "..", "data/sample_submission.csv"))
test_transaction = pd.read_csv(os.path.join(
    package_dir, "..", "data/test_transaction.csv"))
data = test_transaction
data = data[feature_names]
data["ProductCD"] = list(
    map(lambda x: productCD_categories.index(x),  data["ProductCD"]))

predictions = []
for idx in tqdm(range(len(data))):
    request_data = {}
    for feature_name in feature_names:
        request_data[feature_name] = data[feature_name][idx]
    
    r = requests.post("http://localhost:8000/model:predict", data=json.dumps(request_data, cls=NpEncoder), headers=headers)
    predictions.append(r.json()["prediction"])
sample_submission["isFraud"] = predictions
sample_submission.to_csv("/tmp/submission.csv")

"""TODO multuprocessing """
# # split data for the multiprocessing
# data_split = {}
# portion_size = len(data) // num_cpu +1
# start_index = 0
# for split_index in range(num_cpu-1):
#     end_index = start_index + portion_size
#     data_split[split_index] = data.iloc[start_index:end_index].reset_index(drop=True)
#     start_index = end_index
# data_split[num_cpu] = data.iloc[start_index:].reset_index(drop=True)
# del data
# gc.collect()

# def get_perdictions(idx, data, return_dict):
#     predictions = []
#     for idx in tqdm(range(len(data))):
#         request_data = {}
#         for feature_name in feature_names:
#             request_data[feature_name] = data[feature_name][idx]
        
#         r = requests.post("http://localhost:8000/model:predict", data=json.dumps(request_data, cls=NpEncoder), headers=headers)
#         predictions.append(r.json()["prediction"])
#         return_dict[idx] = predictions
#     return return_dict

# # multiprocessing
# manager = multiprocessing.Manager()
# return_dict = manager.dict()
# jobs = []
# for key in data_split.keys():
#     p = multiprocessing.Process(target=get_perdictions, args=(key, data_split[key], return_dict))
#     jobs.append(p)
#     p.start()
# for proc in jobs:
#     proc.join()

# for key in return_dict.keys():
#     print(len(return_dict[key]))

# # write file
# predictions = []
# for idx in range(num_cpu):
#     predictions += return_dict[idx]
# sample_submission["isFraud"] = predictions
# sample_submission.to_csv("/tmp/submission.csv")




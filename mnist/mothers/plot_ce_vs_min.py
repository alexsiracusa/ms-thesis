from mnist.datasets import datasets
import matplotlib.pyplot as plt
import numpy as np
import json
import itertools


data_dir = '../data'
include = list(datasets.keys())[:-3]
noise_types = ['sparse_perlin']

ce_losses = []
min_losses = []

for dataset_name, noise in itertools.product(include, noise_types):
    data_file = f'{data_dir}/{dataset_name}/{noise}.txt'
    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    features_file = f'{data_dir}/{dataset_name}/dataset_features.json'
    with open(features_file, 'r') as f:
        dataset_features = json.load(f)

    min_loss = np.array([data['test_loss'] for data in dataset]).min()
    ce_loss = dataset_features['ce_loss']

    min_losses.append(min_loss)
    ce_losses.append(ce_loss)


plt.scatter(ce_losses, min_losses)
plt.xlabel('CE Loss')
plt.ylabel('Min Test Loss')
plt.show()
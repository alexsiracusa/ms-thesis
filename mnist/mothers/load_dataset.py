import numpy as np
import json
import torch
import itertools
from mnist.datasets import datasets

def load_dataset(
        include,
        noise_types,
        feature_set,
        dataset_feature_set,
        target,
        min_cut_off=0,
        max_cut_off=1,
        max_target=float('inf'),
        data_dir='../data'
):
    features = []
    targets = []

    for dataset_name, noise in itertools.product(include, noise_types):
        data_file = f'{data_dir}/{dataset_name}/{noise}.txt'
        with open(data_file, 'r') as f:
             dataset = [json.loads(line) for line in f]

        features_file = f'{data_dir}/{dataset_name}/dataset_features.json'
        with open(features_file, 'r') as f:
             dataset_features = json.load(f)


        for data in dataset:
            if not min_cut_off <= data['average_density'] <= max_cut_off:
                continue
            if data[target] > max_target:
                continue

            targets.append(data[target])
            data_features = []
            for feature in feature_set:
                data_features += np.array(data[feature]).flatten().tolist()

            for feature in dataset_feature_set:
                data_features += [dataset_features[feature]]

            features.append(data_features)

    return torch.tensor(features), torch.tensor(targets)


if __name__ == '__main__':
    features, targets = load_dataset(
        include=['mnist', 'emnist_letters', 'emnist_balanced', 'fashion_mnist', 'kmnist', 'cifar10', 'sign_mnist'],
        noise_types=['sparse_random'],
        feature_set=['average_density'],
        dataset_feature_set=['num_classes', 'ce_loss'],
        target='test_loss'
    )

    print(features.shape)
    print(targets.shape)

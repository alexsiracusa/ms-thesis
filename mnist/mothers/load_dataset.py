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
        target='test_loss',
        min_cut_off=0,
        max_cut_off=1,
        clip_max_loss=True,
        normalize_loss=False,
        data_dir='../data'
):
    all_features = []
    all_targets = []
    all_jsons = []

    for dataset_name, noise in itertools.product(include, noise_types):
        data_file = f'{data_dir}/{dataset_name}/{noise}.txt'
        with open(data_file, 'r') as f:
             dataset = [json.loads(line) for line in f]

        features_file = f'{data_dir}/{dataset_name}/dataset_features.json'
        with open(features_file, 'r') as f:
             dataset_features = json.load(f)

        max_target = -np.log(1 / datasets[dataset_name]) + 0.05 if clip_max_loss else float('inf')

        features = []
        targets = []
        jsons = []

        for data in dataset:
            if not min_cut_off <= data['average_density'] <= max_cut_off:
                continue
            if data[target] > max_target:
                continue

            jsons.append(data)
            targets.append(data[target])

            data_features = []
            for feature in feature_set:
                data_features += np.array(data[feature]).flatten().tolist()
            for feature in dataset_feature_set:
                data_features += [dataset_features[feature]]

            features.append(data_features)

        # normalize if specified
        if normalize_loss:
            targets = np.array(targets)
            # targets = (targets - targets.min()) / (targets.max() - targets.min())
            targets = (targets - dataset_features['nn_loss_0.5_test']) / (dataset_features['loss_uniform'] - dataset_features['nn_loss_0.5_test'])
            targets = targets.tolist()

        all_features += features
        all_targets += targets
        all_jsons += jsons

    return torch.tensor(all_features), torch.tensor(all_targets), all_jsons


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    include = set(datasets.keys()) - {'sign_mnist', 'path_mnist'}

    features, targets, jsons = load_dataset(
        include=include,
        noise_types=['sparse_random'],
        feature_set=['average_density'],
        dataset_feature_set=['num_classes', 'lr_loss'],
        normalize_loss=True,
    )

    print(features.shape)
    print(targets.shape)
    print(len(jsons))

    plt.scatter(features[:, 0], targets)
    plt.show()

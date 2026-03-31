import torch
import json
import math
import numpy as np
import matplotlib.pyplot as plt


from mnist.mothers.feed_forward import model
from mnist.datasets import datasets, load_parquet
from mnist.util import train_mnist, create_model, sparse_perlin, flatten_images, sparse_random
from mnist.util.sizes import num_input, num_blocks
from torch.utils.data import DataLoader, TensorDataset
from mnist.util import train_mnist, create_model, sparse_perlin, flatten_images, sparse_random, get_num_trainable



mother = model
mother.load_state_dict(torch.load("../mothers/feed_forward_new.pth", map_location="cpu"))

for param in mother.parameters():
    param.requires_grad = False


def generate_children(dataset_name):
    input_file = f'../data/{dataset_name}/sparse_perlin.txt'
    output_file = f'../data/{dataset_name}/sparse_perlin_generated.txt'
    features_file = f'../data/{dataset_name}/dataset_features.json'

    with open(input_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    with open(features_file, 'r') as f:
        features = json.load(f)

    dataset_features = ['nn_loss_0.1_test', 'nn_loss_0.25_test', 'nn_loss_0.5_test']
    # dataset_features = ['lr_loss']
    dataset_features= torch.tensor([features[feature] for feature in dataset_features])

    train_images, train_labels = load_parquet(f"../datasets/parquets/{dataset_name}/train.parquet")
    test_images, test_labels = load_parquet(f"../datasets/parquets/{dataset_name}/train.parquet")

    train_images = flatten_images(train_images, kernel_size=(10, 10), stride=10, padding=0)
    test_images = flatten_images(test_images, kernel_size=(10, 10), stride=10, padding=0)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    batch_size = math.ceil(len(train_dataset) / 200)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for _ in range(20):
        fig, axes = plt.subplots(2, 1, figsize=(10, 5))

        while True:
            data = np.random.choice(dataset)
            if data['average_density'] >= 0.33:
                break

        x = torch.tensor(data['density_map']).flatten()
        x = torch.concatenate((x, dataset_features))
        x.requires_grad_()

        axes[0].imshow(x[:-len(dataset_features)].reshape(11, 36).detach().numpy(), cmap='gray', vmin=0, vmax=1)

        for _ in range(200):
            pred = mother(x)
            pred.backward()

            with torch.no_grad():
                x -= 1 * x.grad
                x.grad.zero_()
                # x = 0.999 * x
                x = x.clamp(min=0, max=1).requires_grad_(True)

                # reset dataset features
                x[-len(dataset_features):] = dataset_features.clone()
                x = x.clone().requires_grad_(True)

        x = x.clamp(min=0, max=1)
        x[-len(dataset_features):] = dataset_features.clone()
        generated = x[:-len(dataset_features)].reshape(11, 36).detach().numpy()
        axes[1].imshow(generated, cmap='gray', vmin=0, vmax=1)
        plt.show()

        # Get real loss
        densities = np.array(generated)
        model = create_model(densities)
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

        train_loss, test_loss = train_mnist(
            model, train_loader, test_loader,
            device=device,
            epochs=5,
            output_size=features['num_classes'],
        )

        data = {
            "original_density_map": data['density_map'],
            "generated_density_map": generated.tolist(),

            "original_trainable": get_num_trainable(data['density_map']),
            "generated_trainable": get_num_trainable(generated.tolist()),

            "original_train_loss": data['train_loss'],
            "original_test_loss": data['test_loss'],
            "original_pred": mother(torch.concatenate((torch.tensor(data['density_map']).flatten(), dataset_features)).flatten()).item(),

            "generated_train_loss": train_loss,
            "generated_test_loss": test_loss,
            "generated_pred": mother(x).item(),
        }

        with open(output_file, 'a') as f:
            f.write(json.dumps(data) + '\n')


if __name__ == "__main__":
    generate_children(
        dataset_name='blood_mnist',
    )

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_parquet(file_path):
    df = pd.read_parquet(file_path)
    labels = df["label"].to_numpy()
    images = df.drop(columns=["label"]).to_numpy().reshape(-1, 1, 50, 50)
    return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


if __name__ == "__main__":
    # from datasets import datasets
    #
    # for dataset, num_classes in datasets.items():
    #     train_images, train_labels = load_parquet(f"./parquets/{dataset}/train.parquet")
    #     actual_classes = len(np.unique(train_labels))
    #     if actual_classes != num_classes:
    #         print(f'Error: {dataset} {actual_classes} {num_classes}')

    train_images, train_labels = load_parquet("./parquets/emnist_balanced/train.parquet")

    fig, axes = plt.subplots(5, 5, figsize=(5, 5), facecolor="black")

    for _, ax in enumerate(axes.flat):
        i = torch.randint(0, train_images.shape[0], (1,))
        ax.imshow(train_images[i].view(50, 50).numpy(), cmap='gray')
        ax.text(
            0, 0, f"{train_labels[i].item()}",
            color='red', fontsize=8, weight='bold', ha='left', va='top'
        )
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(train_images.shape, train_labels.shape)
    print(np.sort(np.unique(train_labels)))

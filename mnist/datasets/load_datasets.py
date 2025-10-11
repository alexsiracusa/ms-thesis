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
    train_images, train_labels = load_parquet("./parquets/not_mnist/train.parquet")
    # test_images, test_labels = load_parquet("./parquets/mnist/test.parquet")

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

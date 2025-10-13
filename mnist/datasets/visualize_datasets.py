from mnist.datasets.load_datasets import load_parquet
from mnist.datasets import datasets
import matplotlib.pyplot as plt
import numpy as np


def display_datasets():
    images = []

    for dataset, num_classes in datasets.items():
        train_images, train_labels = load_parquet(f"./parquets/{dataset}/train.parquet")
        i = np.random.randint(0, train_images.shape[0])
        images.append(train_images[i])


    fig, axes = plt.subplots(5, 5, figsize=(10, 10), facecolor="black")

    for image, dataset, ax in zip(images, datasets.keys(), axes.flat):
        ax.imshow(image.view(50, 50).numpy(), cmap='gray')
        ax.text(
            0, 0, f"{dataset}",
            color='red', fontsize=10, weight='bold', ha='left', va='top'
        )
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    display_datasets()
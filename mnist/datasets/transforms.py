import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from functools import partial


def binary_target_transform(label):
    return int(label.sum() > 0)

transform_list = [
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    # transforms.Lambda(partial(flatten_image, kernel_size=(10, 10), stride=10, padding=0))
]

emnist_transform_list = [
    transforms.RandomRotation((-90, -90)),
    transforms.RandomHorizontalFlip(p=1),
]

composed_transforms = transforms.Compose(transform_list)
emnist_transforms = transforms.Compose(transform_list + emnist_transform_list)

def torch_dataset_to_df(dataset):
    images = torch.stack([img for img, _ in dataset])
    labels = torch.tensor([label for _, label in dataset])

    images_flat = images.view(images.size(0), -1)

    df = pd.DataFrame(images_flat, columns=[f"x{i}" for i in range(50 * 50)])
    df["label"] = labels
    return df


def upscale_df(df):
    labels = df["label"].values
    images = df.drop(columns=["label"]).values  # shape: (N, 784)

    size = int(np.sqrt(images.shape[1]))
    images = np.array(images, dtype=float).reshape(-1, 1, size, size)

    return upscale_to_df(images, labels)

def upscale_to_df(images, labels, size=(50, 50)):
    resize_transform = transforms.Resize(size)
    upscaled_images = torch.stack([resize_transform(torch.tensor(img)) for img in images])
    upscaled_flat = upscaled_images.view(upscaled_images.size(0), -1)

    df = pd.DataFrame(upscaled_flat.numpy(), columns=[f"x{i}" for i in range(50 * 50)])
    df["label"] = labels
    return df

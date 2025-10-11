import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from mnist.datasets.transforms import (
    composed_transforms,
    emnist_transforms,
    upscale_df,
    torch_dataset_to_df,
    upscale_to_df,
    binary_target_transform,
)

import os
import inspect
from PIL import Image
from sklearn.model_selection import train_test_split

from torchvision.datasets import MNIST, KMNIST, FashionMNIST, EMNIST, CIFAR10, ImageFolder
from medmnist import (
    PathMNIST,
    ChestMNIST,
    DermaMNIST,
    OCTMNIST,
    PneumoniaMNIST,
    RetinaMNIST,
    BreastMNIST,
    BloodMNIST,
    TissueMNIST,
    OrganAMNIST,
    OrganCMNIST,
    OrganSMNIST,
)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re

data_folder = './data'

# -----------------------
# CREATE PYTORCH DATASETS
# -----------------------
def create_mnist(output_dir, version, **kwargs):
    os.makedirs(output_dir, exist_ok=True)

    sig = inspect.signature(version.__init__)
    valid_params = {k: v for k, v in kwargs.items() if k in sig.parameters}
    composed = composed_transforms if version != EMNIST else emnist_transforms

    train_dataset = version(root='./data', train=True, transform=composed, download=True, **valid_params)
    test_dataset = version(root='./data', train=False, transform=composed, download=True, **valid_params)

    train_df = torch_dataset_to_df(train_dataset)
    test_df = torch_dataset_to_df(test_dataset)

    pq.write_table(pa.Table.from_pandas(train_df), f"{output_dir}/train.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{output_dir}/test.parquet")


# --------------------------
# CREATE DOWNLOADED DATASETS
# --------------------------
def create_sign_mnist(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # download and unzip original dataset from here and move to your data folder renamed as 'sign_mnist'
    # https://www.kaggle.com/datasets/datamunge/sign-language-mnist?resource=download
    train_df = pd.read_csv(f'{data_folder}/sign_mnist/sign_mnist_train.csv')
    test_df = pd.read_csv(f'{data_folder}/sign_mnist/sign_mnist_test.csv')

    train_df = upscale_df(train_df)
    test_df = upscale_df(test_df)

    pq.write_table(pa.Table.from_pandas(train_df), f"{output_dir}/train.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{output_dir}/test.parquet")


def create_chinese_mnist(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # download and unzip original dataset from here and move to your data folder renamed as 'chinese_mnist'
    # https://data.ncl.ac.uk/articles/dataset/Handwritten_Chinese_Numbers/10280831/1
    folder = f'{data_folder}/chinese_mnist/Raw Dataset/'

    pattern = re.compile(r"Locate\{[^,]+,[^,]+,(\d+)\}\.jpg")

    data = []
    labels = []

    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            label = int(match.group(1)) - 1
            filepath = os.path.join(folder, filename)
            img = Image.open(filepath).convert("L")  # grayscale
            img = img.resize((50,50))  # optional, to ensure consistent size
            pixels = np.array(img).flatten()
            data.append(pixels)
            labels.append(label)

    df = pd.DataFrame(data, columns=[f"x{i}" for i in range(len(data[0]))])
    df.insert(0, "label", labels)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    pq.write_table(pa.Table.from_pandas(train_df), f"{output_dir}/train.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{output_dir}/test.parquet")


def create_kannada_mnist(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # download and unzip original dataset from here and move to your data folder renamed as 'kannada_mnist'
    # https://www.kaggle.com/datasets/higgstachyon/kannada-mnist
    path = f'{data_folder}/kannada_mnist/Kannada_MNIST_npz'

    train_x = np.load(f'{path}/Kannada_MNIST/X_kannada_MNIST_train.npz')['arr_0'].reshape(-1, 1, 28, 28)
    train_y = np.load(f'{path}/Kannada_MNIST/y_kannada_MNIST_train.npz')['arr_0']

    test_x = np.load(f'{path}/Kannada_MNIST/X_kannada_MNIST_test.npz')['arr_0'].reshape(-1, 1, 28, 28)
    test_y = np.load(f'{path}/Kannada_MNIST/y_kannada_MNIST_test.npz')['arr_0']

    train_df = upscale_to_df(train_x, train_y)
    test_df = upscale_to_df(test_x, test_y)

    pq.write_table(pa.Table.from_pandas(train_df), f"{output_dir}/train.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{output_dir}/test.parquet")


def create_dig_mnist(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # download and unzip original dataset from here and move to your data folder renamed as 'kannada_mnist'
    # https://www.kaggle.com/datasets/higgstachyon/kannada-mnist
    path = f'{data_folder}/kannada_mnist/Kannada_MNIST_npz'

    images = np.load(f'{path}/Dig_MNIST/X_dig_MNIST.npz')['arr_0'].reshape(-1, 1, 28, 28)
    labels = np.load(f'{path}/Dig_MNIST/y_dig_MNIST.npz')['arr_0']

    df = upscale_to_df(images, labels)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    pq.write_table(pa.Table.from_pandas(train_df), f"{output_dir}/train.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{output_dir}/test.parquet")


def create_overhead_mnist(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # download and unzip original dataset from here and move to your data folder renamed as 'overhead_mnist'
    # https://www.kaggle.com/datasets/datamunge/overheadmnist/
    train_dataset = ImageFolder(root=f'{data_folder}/overhead_mnist/overhead/training', transform=composed_transforms)
    test_dataset = ImageFolder(root=f'{data_folder}/overhead_mnist/overhead/testing', transform=composed_transforms)

    train_df = torch_dataset_to_df(train_dataset)
    test_df = torch_dataset_to_df(test_dataset)

    pq.write_table(pa.Table.from_pandas(train_df), f"{output_dir}/train.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{output_dir}/test.parquet")


def create_simpsons_mnist(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # download and unzip original dataset from here and move to your data folder renamed as 'overhead_mnist'
    # https://github.com/alvarobartt/simpsons-mnist
    train_dataset = ImageFolder(root=f'{data_folder}/simpsons_mnist/train', transform=composed_transforms)
    test_dataset = ImageFolder(root=f'{data_folder}/simpsons_mnist/test', transform=composed_transforms)

    train_df = torch_dataset_to_df(train_dataset)
    test_df = torch_dataset_to_df(test_dataset)

    pq.write_table(pa.Table.from_pandas(train_df), f"{output_dir}/train.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{output_dir}/test.parquet")


def create_not_mnist(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # this requires manually deleting some corrupted files from the dataset
    # download and unzip original dataset from here and move to your data folder renamed as 'notMNIST'
    # https://www.kaggle.com/datasets/lubaroli/notmnist
    dataset = ImageFolder(root=f'{data_folder}/notMNIST', transform=composed_transforms)

    df = torch_dataset_to_df(dataset)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    pq.write_table(pa.Table.from_pandas(train_df), f"{output_dir}/train.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{output_dir}/test.parquet")


# -------------------------
# CREATE MED_MNIST DATASETS
# -------------------------
def create_med_mnist(output_dir, version, **kwargs):
    os.makedirs(output_dir, exist_ok=True)

    sig = inspect.signature(version.__init__)
    valid_params = {k: v for k, v in kwargs.items() if k in sig.parameters}

    train_dataset = version(root='./data', split="train", size=128, download=True, transform=composed_transforms, **valid_params)
    val_dataset = version(root='./data', split="val", size=128, download=True, transform=composed_transforms, **valid_params)
    test_dataset = version(root='./data', split="test", size=128, download=True, transform=composed_transforms, **valid_params)
    test_dataset = ConcatDataset([val_dataset, test_dataset])

    train_df = torch_dataset_to_df(train_dataset)
    test_df = torch_dataset_to_df(test_dataset)

    pq.write_table(pa.Table.from_pandas(train_df), f"{output_dir}/train.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{output_dir}/test.parquet")



if __name__ == '__main__':
    # Load from torchvision.datasets (the easy ones)
    # create_mnist('parquets/mnist', version=MNIST)
    # create_mnist('parquets/kmnist', version=KMNIST)
    # create_mnist('parquets/fashion_mnist', version=FashionMNIST)
    # create_mnist('parquets/emnist_letters', version=EMNIST, split='letters')
    # create_mnist('parquets/emnist_balanced', version=EMNIST, split='balanced')
    # create_mnist('parquets/cifar10', version=CIFAR10)

    # Load from downloaded folders (need to manually download originals before processing)
    # create_sign_mnist('parquets/sign_mnist')
    # create_chinese_mnist('parquets/chinese_mnist')
    # create_kannada_mnist('parquets/kannada_mnist')
    # create_dig_mnist('parquets/dig_mnist')
    # create_overhead_mnist('parquets/overhead_mnist')
    # create_simpsons_mnist('parquets/simpsons_mnist')
    # create_not_mnist('parquets/not_mnist')  # this requires manually deleting some corrupted files from the dataset

    # Load from MedMNIST datasets
    # create_med_mnist('parquets/path_mnist', version=PathMNIST)
    # create_med_mnist('parquets/chest_mnist', version=ChestMNIST, target_transform=binary_target_transform)
    # create_med_mnist('parquets/dermal_mnist', version=DermaMNIST)
    # create_med_mnist('parquets/oct_mnist', version=OCTMNIST)
    # create_med_mnist('parquets/pneumonia_mnist', version=PneumoniaMNIST)
    # create_med_mnist('parquets/retina_mnist', version=RetinaMNIST)
    # create_med_mnist('parquets/breast_mnist', version=BreastMNIST)
    # create_med_mnist('parquets/blood_mnist', version=BloodMNIST)
    # create_med_mnist('parquets/tissue_mnist', version=TissueMNIST)
    # create_med_mnist('parquets/organ_a_mnist', version=OrganAMNIST)
    # create_med_mnist('parquets/organ_c_mnist', version=OrganCMNIST)
    # create_med_mnist('parquets/organ_s_mnist', version=OrganSMNIST)

    pass




import torch
from torchvision import transforms
from functools import partial
from torchvision.datasets import MNIST, KMNIST, FashionMNIST
from torch.utils.data import DataLoader


def flatten_image(images, kernel_size, **kwargs):
    unfold = torch.nn.Unfold(kernel_size=kernel_size, **kwargs)
    patches = unfold(images)
    patches = patches.transpose(1, 0)
    return patches.flatten()

def flatten_images(images, kernel_size, **kwargs):
    unfold = torch.nn.Unfold(kernel_size=kernel_size, **kwargs)
    patches = unfold(images)
    patches = patches.transpose(1, 2)
    flattened = patches.reshape(images.size(0), -1)
    return flattened


def load_mnist(
        data_folder,
        dataset='MNIST',
        batch_size=64,
        flatten=False,
        shuffle=True,
):
    transform_list = [
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
    ]
    if flatten:
        transform_list += [transforms.Lambda(partial(flatten_image, kernel_size=(10, 10), stride=10, padding=0))]

    transform = transforms.Compose(transform_list)

    dataset = {
        'MNIST': MNIST,
        'KMNIST': KMNIST,
        'FashionMNIST': FashionMNIST
    }[dataset]

    train_dataset = dataset(root=data_folder, train=True, transform=transform, download=True)
    test_dataset = dataset(root=data_folder, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = load_mnist('../../data', dataset='FashionMNIST')
    print(len(train_loader.dataset), len(test_loader.dataset))


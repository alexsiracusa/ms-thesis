from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def load_cifar(data_folder, batch_size=64, shuffle=True):
    transform_list = [
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
    ]

    transform = transforms.Compose(transform_list)

    train_dataset = CIFAR10(root=data_folder, train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root=data_folder, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

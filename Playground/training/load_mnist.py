from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def load_mnist(data_folder, flat=False):
    transform_list = [
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
    ]

    if flat:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    train_dataset = MNIST(root=data_folder, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=data_folder, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

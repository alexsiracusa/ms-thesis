import torch
from util import train_cifar, load_cifar



model = torch.nn.Sequential(
    torch.nn.Linear(7500, 1100),
    torch.nn.ReLU(),
    torch.nn.Linear(1100, 550),
    torch.nn.ReLU(),
    torch.nn.Linear(550, 275),
    torch.nn.ReLU(),
    torch.nn.Linear(275, 10),
)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))
print(0.9e7)

batch_size = 128
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
train_loader, test_loader = load_cifar('../data', batch_size=batch_size)

train_losses, test_losses, epoch_losses = train_cifar(
    model, train_loader, test_loader,
    device=device,
    epochs=epochs,
)

print(train_losses)
print(test_losses)
print(epoch_losses)


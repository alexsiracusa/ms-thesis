import torch
from torch import nn
import time
import numpy as np
import matplotlib.pyplot as plt


def train(
    model: nn.Module,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device=torch.device('cpu'),
    epochs=1
):
    losses = []
    forward_times = []
    backward_times = []

    for epoch in range(epochs):
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            # FORWARD PASS
            start = time.time()  # TIMER START
            output = model.forward(data)
            forward_times.append(time.time() - start)  # TIMER END

            # BACKWARD PASS
            start = time.time()  # TIMER START
            loss = criterion(output[4], labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            backward_times.append(time.time() - start)  # TIMER END

            print(f'{losses[-1]:.3f}  {forward_times[-1]:.3f}  {backward_times[-1]:.3f}')

        if (epoch - 1) % 1 == 0:
            print(f'Loss: {sum(losses[-len(train_loader):]) / len(train_loader)}')

    return losses, forward_times, backward_times
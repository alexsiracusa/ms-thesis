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
    epochs=1,
    print_every_nth_batch=None
):
    model = model.to(device)
    criterion = criterion.to(device)

    losses = []
    forward_times = []
    backward_times = []

    for epoch in range(epochs):
        batch = 0

        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            batch += 1

            # FORWARD PASS
            start = time.time()  # TIMER START
            output = model.forward(data)
            forward_times.append(time.time() - start)  # TIMER END

            # BACKWARD PASS
            start = time.time()  # TIMER START
            loss = criterion(output, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            backward_times.append(time.time() - start)  # TIMER END

            if print_every_nth_batch is not None and (batch - 1) % print_every_nth_batch == 0:
                print(f'{losses[-1]:.3f}  {forward_times[-1]:.3f}  {backward_times[-1]:.3f}  {batch}/{len(train_loader)}')

        if (epoch - 1) % 1 == 0:
            print(f'Loss: {sum(losses[-len(train_loader):]) / len(train_loader)}')

    return losses, forward_times, backward_times
import torch
import torch.nn as nn


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs=1,
    print_batch=None,
    print_epoch=1,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    model = model.to(device)
    criterion = criterion.to(device)

    losses = []

    for epoch in range(epochs):
        for batch, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            output = model.forward(data)
            loss = criterion(output, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if print_batch is not None and batch % print_batch == 0:
                print(f'Batch {batch}/{len(train_loader)}: {losses[-1]:.3f}')

        if print_epoch is not None and epoch % print_epoch == 0:
            print(f'Epoch: {sum(losses[-len(train_loader):]) / len(train_loader)}')

    return model, losses
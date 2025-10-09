import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from cifar10.mother_models.feed_forward_nn import model
from cifar10.util.load_cifar import load_cifar
from cifar10.util import input_sizes, hidden_sizes, output_sizes
from cifar10.util.train import train_cifar

model.load_state_dict(torch.load("../mother_models/feed_forward.pth"))

for param in model.parameters():
    param.requires_grad = False


def generate_children(input_file, output_file, data_folder):
    with open(input_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    train_loader, test_loader = load_cifar(data_folder, batch_size=128, shuffle=False)

    for _ in range(10):
        fig, axes = plt.subplots(2, 1, figsize=(10, 5))

        data = np.random.choice(dataset)
        x = torch.tensor(data['densities']).flatten()
        x.requires_grad_()

        axes[0].imshow(x.reshape(45, 145).detach().numpy(), cmap='gray')

        for _ in range(1000):
            pred = model(x)
            pred.backward()

            with torch.no_grad():
                x -= 10 * x.grad
                x.grad.zero_()

        x = x.clamp(min=0, max=1)
        generated = x.reshape(45, 145).detach().numpy()
        axes[1].imshow(generated, cmap='gray')
        plt.show()

        # Get real loss
        num_iterations = 4
        rows, cols = generated.shape
        densities = np.vstack([np.zeros((cols - rows, cols)), generated])

        train_loss, test_loss = train_cifar(
            input_sizes, hidden_sizes, output_sizes,
            num_iterations, densities,
            train_loader, test_loader,
            num_epochs=3,
        )

        data = {
            "original": data['densities'],
            "generated": x.detach().numpy().tolist(),

            "original_train_loss": data['train_loss'],
            "original_test_loss": data['test_loss'],
            "original_pred": model(torch.tensor(data['densities']).flatten()).item(),

            "generated_train_loss": train_loss,
            "generated_test_loss": test_loss,
            "generated_pred": model(x).item(),
        }

        with open(output_file, 'a') as f:
            f.write(json.dumps(data) + '\n')




if __name__ == "__main__":
    generate_children(
        input_file='../train_epoch=3/perlin_data.txt',
        output_file='../train_epoch=3/perlin_generated.txt',
        data_folder='../../data',
    )

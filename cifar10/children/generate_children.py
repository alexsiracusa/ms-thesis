import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from cifar10.mothers.feed_forward import model
from cifar10.util import load_cifar, train_cifar, create_model
from cifar10.util import input_sizes, hidden_sizes, output_sizes

model.load_state_dict(torch.load("../mothers/feed_forward.pth"))


for param in model.parameters():
    param.requires_grad = False

def generate_children(input_file, output_file, data_folder):
    with open(input_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    train_loader, test_loader = load_cifar(data_folder, batch_size=128, shuffle=False)

    for _ in range(3):
        fig, axes = plt.subplots(2, 1, figsize=(10, 5))

        data = np.random.choice(dataset)
        x = torch.tensor(data['density_map']).flatten()
        x.requires_grad_()

        axes[0].imshow(x.reshape(11, 36).detach().numpy(), cmap='gray', vmin=0, vmax=1)

        for _ in range(300):
            pred = model(x)
            pred.backward()

            with torch.no_grad():
                x -= 1 * x.grad
                x.grad.zero_()
                # x = 0.99 * x
                x = x.clamp(min=0, max=1).requires_grad_(True)

        x = x.clamp(min=0, max=1)
        generated = x.reshape(11, 36).detach().numpy()
        axes[1].imshow(generated, cmap='gray', vmin=0, vmax=1)
        plt.show()

        # Get real loss
        densities = np.array(generated)
        model = create_model(densities)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_losses, test_losses, epoch_losses = train_cifar(
            model, train_loader, test_loader,
            device=device,
            epochs=5,
        )

        data = {
            "original_density_map": data['densities'],
            "generated_density_map": x.detach().numpy().tolist(),

            "original_train_losses": data['train_losses'],
            "original_test_losses": data['test_losses'],
            "original_pred": model(torch.tensor(data['density_map']).flatten()).item(),

            "generated_train_losses": train_losses,
            "generated_test_losses": test_losses,
            "generated_pred": model(x).item(),
        }

        with open(output_file, 'a') as f:
            f.write(json.dumps(data) + '\n')




if __name__ == "__main__":
    generate_children(
        input_file='../data/sparse_perlin.txt',
        output_file='../data/sparse_perlin_generated.txt',
        data_folder='../../data',
    )

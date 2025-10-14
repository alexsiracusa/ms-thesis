import wandb
import torch
import numpy as np
import random
from cifar10.util import train_cifar, create_model, sparse_perlin, sparse_random, load_cifar
from cifar10.util.sizes import num_input, num_blocks


def wandb_run():
    run = wandb.init()
    noise = run.config.noise
    epochs = run.config.epochs
    batch_size = run.config.batch_size
    run_id = run.config.run_id

    # Load dataset
    train_loader, test_loader = load_cifar('../data', batch_size=batch_size)

    # Pick noise
    if noise == 'sparse_random':
        p_random = 0.99 * random.random() + 0.01
        wandb.log({"p_random": p_random})
        densities = sparse_random((num_blocks - num_input, num_blocks), p_random=p_random)
    elif noise == 'sparse_perlin':
        clip = 0.33
        wandb.log({"clip": clip})
        densities = sparse_perlin((num_blocks - num_input, num_blocks), clip=clip)
    elif noise == 'sparse_perlin_clip':
        clip = 0.99 * random.random() + 0.01
        wandb.log({"clip": clip})
        densities = sparse_perlin((num_blocks - num_input, num_blocks), clip=clip)

    # Train Model
    model = create_model(densities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log model parameters
    wandb.log({"average_density": densities.mean()})
    wandb.run.summary["density_map"] = np.array(densities.tolist())
    wandb.log({"batch_size": batch_size})

    train_losses, test_losses, epoch_losses = train_cifar(
        model, train_loader, test_loader,
        device=device,
        epochs=epochs,
    )

    # Log results
    wandb.log({"train_losses": train_losses})
    wandb.log({"test_losses": test_losses})
    wandb.log({"epoch_losses": epoch_losses})

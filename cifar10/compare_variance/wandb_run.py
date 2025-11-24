import wandb
import torch
import numpy as np
from cifar10.util import train_cifar, create_model, sparse_perlin, sparse_random, load_cifar
from cifar10.util import num_iterations
from Sequential2D.util import build_sequential2d


def wandb_run():
    run = wandb.init()
    noise = run.config.noise
    block_size = run.config.block_size
    epochs = run.config.epochs
    batch_size = run.config.batch_size
    run_id = run.config.run_id

    # Compute blocks
    num_input = int(7500 / block_size)
    num_hidden = int(3000 / block_size)
    num_output = 1
    num_blocks = num_input + num_hidden + num_output
    sizes = [block_size] * (num_input + num_hidden) + [10]

    # Load dataset
    train_loader, test_loader = load_cifar('../data', batch_size=batch_size)

    # Pick noise
    p_random = 0.66
    wandb.log({"p_random": p_random})
    densities = sparse_random((num_blocks - num_input, num_blocks), p_random=p_random)

    # Train Model
    model = build_sequential2d(
        sizes,
        type='linear',
        num_input_blocks=num_input,
        num_output_blocks=num_output,
        num_iterations=num_iterations,
        densities=densities,
        weight_init='weighted',
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log model parameters
    wandb.log({"average_density": densities.mean()})
    wandb.run.summary["density_map"] = np.array(densities).tolist()
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

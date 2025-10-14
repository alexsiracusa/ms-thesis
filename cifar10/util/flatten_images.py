import torch

def flatten_images(images, kernel_size, **kwargs,):
    unfold = torch.nn.Unfold(kernel_size=kernel_size, **kwargs)

    patches = unfold(images)                    # (batch_size, num_channels * ∏(kernel_size), L)
    patches = patches.transpose(2, 1)           # (batch_size, L, num_channels * ∏(kernel_size))
    flat_images = patches.flatten(start_dim=1)  # (batch_size, *)

    return flat_images



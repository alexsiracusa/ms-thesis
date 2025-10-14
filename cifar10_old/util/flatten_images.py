
"""
This will flatten a batch of images keeping the values with each `kernel_size` 'patch' near each other

Args:
    images (batch_size, num_channels, width, height): The images to flatten
    kernel_size (int or tuple): The dimensions of the sliding window
    **kwargs: dilation, padding, stride.  The parameters accepted by `torch.unfold()`

Returns
    flat_images (batch_size,
"""
import torch.nn


def flatten_images(
        images,
        kernel_size,
        **kwargs,
):
    unfold = torch.nn.Unfold(kernel_size=kernel_size, **kwargs)

    patches = unfold(images)                    # (batch_size, num_channels * ∏(kernel_size), L)
    patches = patches.transpose(2, 1)           # (batch_size, L, num_channels * ∏(kernel_size))
    flat_images = patches.flatten(start_dim=1)  # (batch_size, *)

    return flat_images


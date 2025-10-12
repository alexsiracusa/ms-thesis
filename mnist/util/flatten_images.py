import torch

def flatten_image(images, kernel_size, **kwargs):
    unfold = torch.nn.Unfold(kernel_size=kernel_size, **kwargs)
    patches = unfold(images)
    patches = patches.transpose(1, 0)
    return patches.flatten()

def flatten_images(images, kernel_size, **kwargs):
    unfold = torch.nn.Unfold(kernel_size=kernel_size, **kwargs)
    patches = unfold(images)
    patches = patches.transpose(1, 2)
    flattened = patches.reshape(images.size(0), -1)
    return flattened

if __name__ == "__main__":
    image = torch.tensor([
        [ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=torch.float32).unsqueeze(0)

    print(flatten_image(image, kernel_size=(2, 2), stride=2, padding=0))

    print(flatten_images(torch.cat([image, image]).unsqueeze(1), kernel_size=(2, 2), stride=2, padding=0))

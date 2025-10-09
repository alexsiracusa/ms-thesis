import matplotlib.pyplot as plt
import numpy as np
import json

from cifar10.util import get_num_trainable


data_file='../train_epoch=3/perlin_generated.txt'

with open(data_file, 'r') as f:
    dataset = [json.loads(line) for line in f]

plt.axis('off')

for i, child in enumerate(dataset):
    densities = np.array(child['generated']).reshape(45, 145)
    original = np.array(child['original']).reshape(45, 145)

    if get_num_trainable(densities) < 2e6:
        densities = densities.clip(min=0, max=1)
        # print(np.min(densities), np.max(densities))

        plt.imshow(densities, cmap='gray', vmin=0, vmax=1)
        plt.savefig(f'child_{i}.png', bbox_inches='tight', pad_inches=0)

        # print(np.min(original), np.max(original))
        plt.imshow(original, cmap='gray', vmin=0, vmax=1)
        plt.savefig(f'original_{i}.png', bbox_inches='tight', pad_inches=0)

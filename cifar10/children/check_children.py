import matplotlib.pyplot as plt
import numpy as np
import json

from cifar10_old.util import get_num_trainable


data_file='../data/sparse_perlin_generated.txt'

with open(data_file, 'r') as f:
    dataset = [json.loads(line) for line in f]

plt.axis('off')

for i, child in enumerate(dataset):
    densities = np.array(child['generated_density_map']).reshape(11, 36)
    original = np.array(child['original_density_map']).reshape(11, 36)

    plt.imshow(densities, cmap='gray', vmin=0, vmax=1)
    plt.savefig(f'child_{i}.png', bbox_inches='tight', pad_inches=0)

    plt.imshow(original, cmap='gray', vmin=0, vmax=1)
    plt.savefig(f'original_{i}.png', bbox_inches='tight', pad_inches=0)

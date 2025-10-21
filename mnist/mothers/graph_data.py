from mnist.mothers import load_dataset
import matplotlib.pyplot as plt
from mnist.datasets import datasets
from mnist.util import get_num_trainable


include = set(datasets.keys()) - {'path_mnist', 'sign_mnist'}
# include = ['organ_a_mnist']

params = {
    'include': include,
    'feature_set': ['average_density'],
    'dataset_feature_set': ['ce_loss'],
    'target': 'test_loss',
    # 'clip_max_loss': False,
    'normalize_loss': True,
}

_, targets, jsons = load_dataset(**params, noise_types=['sparse_random'])
_, targets_perlin, jsons_perlin = load_dataset(**params, noise_types=['sparse_perlin'])


# Graph data
trainable_parameters = [get_num_trainable((data['density_map'])) for data in jsons]
# test_losses = ([data['test_loss'] for data in jsons])

plt.scatter(
    trainable_parameters, targets,
    label='Sparse Random',
    alpha=0.5,
    s=5,
)

trainable_parameters = [get_num_trainable((data['density_map'])) for data in jsons_perlin]
# test_losses = ([data['test_loss'] for data in jsons_perlin])

plt.scatter(
    trainable_parameters, targets_perlin,
    label='Sparse Perlin',
    alpha=0.5,
    s=5,
)

plt.xlabel('Num. Trainable Parameters')
plt.ylabel('Test Loss')
plt.legend(loc='upper right')
plt.show()




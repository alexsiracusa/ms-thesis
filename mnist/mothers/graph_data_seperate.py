from mnist.mothers import load_dataset
import matplotlib.pyplot as plt
from mnist.datasets import datasets
from mnist.util import get_num_trainable


include = set(datasets.keys()) - {'path_mnist', 'sign_mnist'}
# include = ['organ_a_mnist']

plt.figure(figsize=(7.5, 5))

for dataset in include:
    params = {
        'include': [dataset],
        'feature_set': ['average_density'],
        'dataset_feature_set': ['lr_loss'],
        'target': 'test_loss',
    }

    _, targets, jsons = load_dataset(**params, noise_types=['sparse_random', 'sparse_perlin'])

    # Graph data
    trainable_parameters = [get_num_trainable((data['density_map'])) for data in jsons]
    plt.scatter(
        trainable_parameters, targets,
        label=f'{dataset}',
        alpha=0.5,
        s=5,
    )


plt.xlabel('Num. Trainable Parameters')
plt.ylabel('Test Loss')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3.0, frameon=False)
plt.tight_layout()
plt.show()




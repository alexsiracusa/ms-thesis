from generate_data import generate_data
from cifar10.util.random_densities import perlin_densities

generate_data(
    data_folder='../data',
    output_file='./perlin_data.txt',
    density_fn=perlin_densities,
)

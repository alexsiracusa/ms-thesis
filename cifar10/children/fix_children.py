import matplotlib.pyplot as plt
import numpy as np
import json


data_file = '../train_epoch=3/perlin_generated_old.txt'
output_file = '../train_epoch=3/perlin_generated_fixed.txt'

with open(data_file, 'r') as f:
    dataset = [json.loads(line) for line in f]

for i, child in enumerate(dataset):
    data = child
    densities = np.array(child['generated']).reshape(45, 145).clip(0,1)
    data['generated'] = densities.tolist()

    with open(output_file, 'a') as f:
        f.write(json.dumps(data) + '\n')

import json
import numpy as np
from mnist.util import get_num_trainable
import matplotlib.pyplot as plt

with open('train_data.txt', 'r') as f:
    dataset = [json.loads(line) for line in f]

trainable = np.array([get_num_trainable(data['densities']) for data in dataset]).reshape(-1, 25)
losses = np.array([data['test_loss'] for data in dataset]).reshape(-1, 25)

for train, loss in zip(trainable, losses):
    plt.scatter(train, loss)

# plt.ylim(0.25, 0.6)
plt.savefig('fig.png')

print(losses)


import matplotlib.pyplot as plt
import numpy as np
import json


with open("./train_data.txt", 'r') as f:
    data = [json.loads(line) for line in f]

sum_densities = [sum(np.array(data['densities']).flatten()) for data in data]
test_losses = [data['test_loss'] for data in data]

print(sum_densities[:5])
print(test_losses[:5])

plt.scatter(sum_densities, test_losses)
plt.savefig("graph.png")




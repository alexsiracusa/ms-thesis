import json
from cifar10.util import get_num_trainable
import matplotlib.pyplot as plt
import random


with open('./train_epoch=3/perlin_data.txt', 'r') as f:
    data = [json.loads(line) for line in f]

lowest_loss = min(data, key=lambda item: item['test_loss'])
random_sample = random.sample(data, 3)

print(lowest_loss['test_loss'])
print(get_num_trainable(lowest_loss['densities']))


fig, axes = plt.subplots(4, 1, figsize=(16, 4))

plt.axis('off')
axes[0].imshow(lowest_loss['densities'], cmap='gray')
axes[1].imshow(random_sample[0]['densities'], cmap='gray')
axes[2].imshow(random_sample[1]['densities'], cmap='gray')
axes[3].imshow(random_sample[2]['densities'], cmap='gray')

axes[0].text(0, 0, f"{lowest_loss['test_loss']:.3f}", fontsize=8, color='red')
axes[1].text(0, 0, f"{random_sample[0]['test_loss']:.3f}", fontsize=8, color='red')
axes[2].text(0, 0, f"{random_sample[1]['test_loss']:.3f}", fontsize=8, color='red')
axes[3].text(0, 0, f"{random_sample[2]['test_loss']:.3f}", fontsize=8, color='red')

axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')
axes[3].axis('off')

# plt.imshow(lowest_loss['densities'], cmap='gray')
plt.savefig('lowest_loss.png', bbox_inches='tight', pad_inches=0.1)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import random
import json
from cifar10.util import get_num_trainable

random.seed(0)

with open('../train_epoch=3/perlin_data.txt', 'r') as f:
    train_data = [json.loads(line) for line in f]
    random.shuffle(train_data)
    train_cut = int(0.8 * len(train_data))

# X = [np.array(data['densities']).flatten() for data in train_data]
X = [[get_num_trainable(data['densities'])] for data in train_data]
y = [data['test_loss'] for data in train_data]

X_train, y_train = X[:train_cut], y[:train_cut]
X_test, y_test = X[train_cut:], y[train_cut:]

# model = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
model = DecisionTreeRegressor(max_depth=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(mse)

num_trainable = [get_num_trainable(data['densities']) for data in train_data][train_cut:]
plt.scatter(num_trainable, y_test, label='Data points')
plt.scatter(num_trainable, y_pred, label='Predictions')
plt.text(
    1, 1.05, f'Loss: {mse:.7f}',
    transform=plt.gca().transAxes,
    ha="right", va="top",
    fontsize=12, color="red"
)
plt.legend(loc='upper right')
plt.xlabel('Num. Trainable Parameters')
plt.ylabel('Test Loss')
plt.savefig('decision_tree.png')

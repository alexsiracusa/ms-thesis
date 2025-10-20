from cifar10.util import get_num_trainable
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from cifar10.mothers.graphs import test_vs_pred, num_train_vs_test_graph
import json


with open('../data/sparse_perlin.txt', 'r') as f:
    train_data = [json.loads(line) for line in f]

X = [[get_num_trainable(data['density_map'])] for data in train_data]
y = [data['test_losses'][-1] for data in train_data]

X_train, X_test, y_train, y_test, jsons_train, jsons_test = train_test_split(X, y, train_data, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")

num_train_vs_test_graph(
    y_test, y_pred, jsons_test, mse,
    show=True
)

# test_vs_pred(
#     y_test, y_pred, mse, show=True,
#     ylim=(None, None),
#     xlim=(None, None)
# )







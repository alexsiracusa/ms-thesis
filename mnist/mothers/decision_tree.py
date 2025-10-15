from mnist.mothers import load_dataset
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mnist.mothers.graphs import test_vs_pred, num_train_vs_test_graph
from mnist.datasets import datasets


include = list(datasets.keys())[:-3]
super_include = ['blood_mnist', 'chinese_mnist']
include = set(include) - set(super_include)

params = {
    'noise_types': ['sparse_perlin'],
    'feature_set': ['density_map'],
    'dataset_feature_set': ['ce_loss'],
    'target': 'test_loss',
    'min_cut_off': 0,
    'max_cut_off': 1,
    'max_target': 5,
}

features, targets, jsons = load_dataset(**params, include=include)
super_features, super_targets, super_jsons = load_dataset(**params, include=super_include)

X, y = features.numpy(), targets.numpy()
X_train, X_test, y_train, y_test, jsons_train, jsons_test = train_test_split(X, y, jsons, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=8)
model.fit(X_train, y_train)

# Predict and evaluate
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
#
# print(f"Mean Squared Error: {mse:.4f}")
#
# num_train_vs_test_graph(
#     y_test, y_pred, jsons_test, mse,
#     show=True
# )
#
# test_vs_pred(
#     y_test, y_pred, mse, show=True,
#     ylim=(None, None),
#     xlim=(None, None)
# )


# Super test set
y_pred = model.predict(super_features.numpy())
mse = mean_squared_error(y_pred, super_targets.numpy())

num_train_vs_test_graph(
    super_targets.numpy(), y_pred, super_jsons, mse,
    show=True
)

test_vs_pred(
    super_targets.numpy(), y_pred, mse, show=True,
    ylim=(None, None),
    xlim=(None, None)
)






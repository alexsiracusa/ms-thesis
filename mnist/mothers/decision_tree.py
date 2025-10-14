from mnist.mothers import load_dataset
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

features, targets = load_dataset(
    include=['mnist', 'emnist_letters', 'emnist_balanced', 'fashion_mnist', 'kmnist', 'cifar10', 'sign_mnist'],
    noise_types=['sparse_perlin'],
    feature_set=['average_density'],
    dataset_feature_set=['ce_loss'],
    target='test_loss',
    min_cut_off=0,
    max_cut_off=1,
    max_target=5,
)

X, y = features.numpy(), targets.numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=8)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")

plt.scatter(y_test, y_pred)
plt.xlabel('Test Error')
plt.ylabel('Prediction')

# plt.ylim(0, 4)
# plt.xlim(0, 4)
plt.show()






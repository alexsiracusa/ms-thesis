import numpy as np

def generate_linear_data(num_samples, num_features, noise_std=0.1):
    # Generate random features
    X = np.random.rand(num_samples, num_features) * 10

    # Generate random coefficients (true relationship between features and target)
    true_coefficients = np.random.rand(num_features)

    # Generate the target variable with added noise
    y = np.dot(X, true_coefficients) + np.random.normal(0, noise_std, num_samples)
    y = y.reshape(-1, 1)

    return X, y
from mnist.datasets import datasets, load_parquet
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def generate_dataset_features(include):
    for dataset, num_classes in list(datasets.items()):
        # load existing features
        feature_file_path = f'../data/{dataset}/dataset_features.json'
        try:
            with open(feature_file_path, 'r', encoding='utf-8') as f:
                features = json.load(f)
        except:
            features = {}

        train_images, train_labels = load_parquet(f"./parquets/{dataset}/train.parquet")
        test_images, test_labels = load_parquet(f"./parquets/{dataset}/test.parquet")

        train_images = train_images.flatten(start_dim=1)
        test_images = test_images.flatten(start_dim=1)

        if 'linear_regression' in include:
            model = LogisticRegression(max_iter=100)
            model.fit(train_images, train_labels)

            # Predict and compute MSE
            probs = model.predict_proba(test_images)
            ce_loss = log_loss(test_labels, probs)

            features['ce_loss'] = ce_loss
            print(f'{dataset}: {ce_loss}')

        features['num_classes'] = num_classes
        features['loss_uniform'] = -np.log(1 / num_classes)

        # save features
        with open(feature_file_path, 'w') as f:
            json.dump(features, f, indent=4)


if __name__ == "__main__":
    generate_dataset_features(include=['linear_regression'])

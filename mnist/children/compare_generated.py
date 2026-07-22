import json
import statistics
from mnist.mothers import load_dataset

def compare(
    dataset_name,
):
    generated_file = f'../data/{dataset_name}/sparse_perlin_generated.txt'

    with open(generated_file, 'r') as f:
        generated = [json.loads(line) for line in f]

    _, _, original = load_dataset(
        noise_types=['sparse_perlin'],
        feature_set=['density_map'],
        dataset_feature_set=[],
        include=[dataset_name],
        min_cut_off=0.33,
    )

    original_test_losses = [data['test_loss'] for data in original]
    generated_test_losses = [data['generated_test_loss'] for data in generated]

    print(f"""
    original_mean:      {statistics.mean(original_test_losses)}
    original_std:       {statistics.stdev(original_test_losses)}
    original_n:         {len(original_test_losses)}
    
    generated_mean:     {statistics.mean(generated_test_losses)}
    generated_std:      {statistics.stdev(generated_test_losses)}
    generated_n:        {len(generated_test_losses)}
    """)


if __name__ == '__main__':
    compare('blood_mnist')
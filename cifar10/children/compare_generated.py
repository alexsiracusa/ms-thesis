import json
import statistics
from mnist.mothers import load_dataset

def compare():
    original_file = f'../data/sparse_perlin.txt'
    generated_file = f'../data/sparse_perlin_generated.txt'

    with open(original_file, 'r') as f:
        original = [json.loads(line) for line in f]

    with open(generated_file, 'r') as f:
        generated = [json.loads(line) for line in f]

    original_test_losses = [data['test_losses'][-1] for data in original]
    generated_test_losses = [data['generated_test_losses'][-1] for data in generated]

    print(f"""
    original_mean:      {statistics.mean(original_test_losses)}
    original_std:       {statistics.stdev(original_test_losses)}
    
    generated_mean:     {statistics.mean(generated_test_losses)}
    generated_std:      {statistics.stdev(generated_test_losses)}
    """)


if __name__ == '__main__':
    compare()
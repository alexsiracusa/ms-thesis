import json
from generate_graphs import get_num_trainable

with open('./train_epoch=3/train_data.txt', 'r') as f:
    dataset = [json.loads(line) for line in f]

arr



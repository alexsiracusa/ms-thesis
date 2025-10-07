import matplotlib.pyplot as plt
import numpy as np
import json

from cifar10.util import get_num_trainable


def generate_graphs(
    data_file='./train_epoch=3/perlin_generated.txt',
    background_file='./train_epoch=3/perlin_data.txt',
    graph_file='children.png',
):
    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    with open(background_file, 'r') as f:
        background = [json.loads(line) for line in f]

    background_parameters = [get_num_trainable((data['densities'])) for data in background]
    background_losses = [data['test_loss'] for data in background]

    trainable_parameters = [get_num_trainable((data['original'])) for data in dataset]
    test_losses = [data['original_test_loss'] for data in dataset]

    generated_parameters = [get_num_trainable(np.array(data['generated']).reshape(45, 145)) for data in dataset]
    generated_losses = [data['generated_test_loss'] for data in dataset]

    plt.scatter(
        background_parameters, background_losses,
        label='Original',
        alpha=0.1,
        s=5,
    )

    plt.scatter(
        trainable_parameters, test_losses,
        label='Original',
        alpha=1,
        s=15,
    )

    plt.scatter(
        generated_parameters, generated_losses,
        label='Generated',
        alpha=1,
        s=15,
    )

    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.legend(loc='upper right')
    plt.savefig(graph_file)

    # from cifar10.mother_models.feed_forward_nn import model
    # model.load_state_dict(torch.load("./mother_models/feed_forward.pth"))
    #
    # for data in dataset:
    #     print(f"""
    #     ORIGINAL
    #     Trainable: {get_num_trainable(data['original'])}
    #     Predicted: {model}
    #
    #     Generated:
    #
    #     """)


if __name__ == '__main__':
    generate_graphs(
        data_file='./train_epoch=3/perlin_generated.txt',
        graph_file='children.png',
    )




from Sequential2D.RecurrentSequential2D import RecurrentSequential2D, FlatRecurrentSequential2D
import torch.nn.functional as F
import torch

I = torch.nn.Identity()
#          2     2     2     2
blocks = [[I,    None, None, None],  # 2
          [None, I,    None, None],  # 2
          [I,    I,    I,    I   ],  # 2
          [I,    I,    I,    I   ]]  # 2
sizes = [2,  2,  2, 2]

model = RecurrentSequential2D(
    blocks,
    num_iterations=1,
    activations=F.relu
)

# input_seq: (seq_len, num_blocks, num_samples, block_size) = (3, 2, 1, 2)
input_seq = [
    [torch.tensor([[1, 1, 1, 1]]), torch.tensor([[1, 1, 1, 1]])],
    [torch.tensor([[1, 1, 1, 1]]), torch.tensor([[1, 1, 1, 1]])],
    [torch.tensor([[1, 1, 1, 1]]), torch.tensor([[1, 1, 1, 1]])],
]

outputs = model.forward(input_seq)
print(outputs)


model = FlatRecurrentSequential2D(
    blocks,
    sizes,
    num_iterations=1,
    activations=F.relu
)

# (seq_len, batch_size, input_size) = (3, 2, 4)
input_seq = torch.tensor([
    [[1, 1, 1, 1], [1, 1, 1, 1]],
    [[1, 1, 1, 1], [1, 1, 1, 1]],
    [[1, 1, 1, 1], [1, 1, 1, 1]],
])

output_seq = model.forward(input_seq)
print(output_seq)



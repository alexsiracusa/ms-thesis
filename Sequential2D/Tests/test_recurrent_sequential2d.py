from Sequential2D.RecurrentSequential2D import RecurrentSequential2D
import torch.nn.functional as F
import torch

I = torch.nn.Identity()
#          2     2     2     2
blocks = [[I,    None, None, None],  # 2
          [None, I,    None, None],  # 2
          [I,    I,    I,    I   ],  # 2
          [I,    I,    I,    I   ]]  # 2

model = RecurrentSequential2D(
    blocks,
    num_iterations=1,
    activations=F.relu
)

# input_seq: (seq_len, num_blocks, num_samples, block_size) = (3, 2, 1, 2)
input_seq = [
    [torch.tensor([[1, 0]]), torch.tensor([[1, 0]])],
    [torch.tensor([[1, 0]]), torch.tensor([[1, 0]])],
    [torch.tensor([[1, 0]]), torch.tensor([[1, 0]])],
]

outputs = model.forward(input_seq)

for output in outputs:
    for block in output:
        for sample in block:
            print(sample, end=' ')
    print("\n")



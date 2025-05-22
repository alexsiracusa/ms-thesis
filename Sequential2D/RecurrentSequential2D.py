import torch
from torch import nn
from .IterativeSequential2D import IterativeSequential2D, FlatIterativeSequential2D
import torch.nn.functional as F


class RecurrentSequential2D(nn.Module):
    def __init__(
        self,
        blocks,
        num_iterations=1,
        activations=F.relu
    ):
        super(RecurrentSequential2D, self).__init__()

        self.blocks = blocks
        self.num_iterations = num_iterations
        self.sequential = IterativeSequential2D(
            blocks,
            num_iterations=num_iterations,
            activations=activations,
            return_last=False
        )
        self.activations = activations

    """
    NOTE: this is really annoying to use, please don't.  Use the flat version instead
    
    Args:
        input_seq: The input sequences of shape:
            (seq_len, num_blocks, batch_size, block_size)
              'seq_len'    - length of the sequences
              'num_blocks' - number of blocks in sequential2d model
              'batch_size' - number of samples in the mini-batch
              'block_size' - number of output features in each block (inhomogeneous: may be different for each block)

    Returns:
        output_seq: The output sequences of shape:
            (seq_len, num_blocks, batch_size, block_size)
              'seq_len'    - length of the sequences
              'num_blocks' - number of blocks in sequential2d model
              'batch_size' - number of samples in the mini-batch
              'block_size' - number of output features in each block (inhomogeneous: may be different for each block)
    """
    def forward(self, input_seq, hidden=None):
        # input_seq: (seq_len, num_blocks, batch_size, block_size)

        seq_len = len(input_seq)

        # output_seq: (seq_len, num_blocks, batch_size, block_size)
        output_seq = []
        for t in range(seq_len):
            input_t = input_seq[t]  # (num_blocks, batch_size, block_size)
            output_t = output_seq[-1] if output_seq else hidden

            if output_t is not None:
                input_t += output_t[len(input_t):]

            output = self.sequential(input_t)
            output_seq.append(output)

        return output_seq


class FlatRecurrentSequential2D(nn.Module):
    def __init__(
        self,
        blocks,
        sizes,
        num_iterations=1,
        activations=F.relu
    ):
        super(FlatRecurrentSequential2D, self).__init__()

        self.blocks = blocks
        self.sizes = sizes
        self.num_iterations = num_iterations
        self.sequential = FlatIterativeSequential2D(blocks, sizes, num_iterations=num_iterations, activations=activations)
        self.activations = activations

    """
    Args:
        input_seq: The input sequences of shape:
            (seq_len, batch_size, input_size)
              'seq_len'    - length of the sequences
              'batch_size' - number of samples in the mini-batch
              'input_size' - number of input features

    Returns:
        output_seq: The output sequences of shape:
            (seq_len, batch_size, input_size)
              'seq_len'     - length of the sequences
              'batch_size'  - number of samples in the mini-batch
              'output_size' - number of output features
        """
    def forward(self, input_seq):

        seq_len = len(input_seq)

        output_seq = []
        for t in range(seq_len):
            input_t = output_seq[-1] if output_seq else input_seq[t]
            input_t[:, :input_seq[t].size(1)] = input_seq[t]

            output = self.sequential(input_t)
            output_seq.append(output)

        return torch.stack(output_seq)

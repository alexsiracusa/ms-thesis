import torch
from torch import nn
from Sequential2D import IterativeSequential2D, FlatIterativeSequential2D
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
        )
        self.activations = activations

    """
    NOTE: This is really annoying to use as the entire input_seq cannot be a tensor due to the block_size 
          dimension being inhomogeneous, making it hard to reshape properly.  Use the flat version instead.
    
    Parameter Values:
        'seq_len'    - length of the sequences
        'num_blocks' - number of blocks in sequential2d model
        'batch_size' - number of samples in the mini-batch
        'block_size' - number of output features in each block (inhomogeneous: may be different for each block)
    
    Args:
        input_seq (seq_len, num_blocks, batch_size, block_size): The input sequences

    Returns:
        output_seq (seq_len, num_blocks, batch_size, block_size): The output sequences
    """
    def forward(self, input_seq, hidden=None):
        seq_len = len(input_seq)

        output_seq = []  # (seq_len, num_blocks, batch_size, block_size)

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
        self.sequential = FlatIterativeSequential2D(
            blocks,
            sizes,
            num_iterations=num_iterations,
            activations=activations
        )
        self.activations = activations

    """
    Parameter Values:
        'seq_len'    - length of the sequences
        'batch_size' - number of samples in the mini-batch
        'input_size' - number of input features = number of output features
    
    Args:
        input_seq (seq_len, batch_size, input_size): The input sequences
            
    Returns:
        output_seq (seq_len, batch_size, input_size): The output sequences
    """
    def forward(self, input_seq, batch_first=False):
        if batch_first:
            input_seq = input_seq.transpose(0, 1)

        output_seq = []

        for t in range(len(input_seq)):
            if output_seq:
                prev_output = output_seq[-1]

                input_t = torch.cat([
                    input_seq[t],
                    prev_output[:, input_seq[t].size(1):]  # Remaining dimensions if any
                ], dim=1)
            else:
                input_t = input_seq[t]

            output = self.sequential(input_t)
            output_seq.append(output)

        return torch.stack(output_seq)

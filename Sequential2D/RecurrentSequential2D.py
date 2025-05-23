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
                input_t += [torch.detach(t) if t is not None else t for t in output_t[len(input_t):]]

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
    def forward(self, input_seq, batch_first=False):
        if batch_first:
            input_seq = input_seq.transpose(0, 1)

        output = None

        for t in range(len(input_seq)):
            if output is not None:
                input_t = torch.cat([
                    input_seq[t],
                    output[:, input_seq[t].size(1):]  # Remaining dimensions if any
                ], dim=1)
            else:
                input_t = input_seq[t]

            output = self.sequential(torch.detach(input_t))

        return output


    # def forward(self, input_seq, batch_first=False):
    #     if batch_first:
    #         input_seq = input_seq.transpose(0, 1)
    #
    #     output_seq = []
    #
    #     for t in range(len(input_seq)):
    #         if output_seq:
    #             prev_output = torch.detach(output_seq[-1])
    #
    #             input_t = torch.cat([
    #                 input_seq[t],
    #                 prev_output[:, input_seq[t].size(1):]  # Remaining dimensions if any
    #             ], dim=1)
    #         else:
    #             input_t = input_seq[t]
    #
    #         output = self.sequential(torch.detach(input_t))
    #         output_seq.append(output)
    #
    #     output_seq = torch.stack(output_seq)
    #     output_seq = torch.transpose(output_seq, 0, 1)
    #     return output_seq



    # def forward(self, input_seq, batch_first=False):
    #     if batch_first:
    #         input_seq = input_seq.transpose(0, 1)
    #
    #     seq_len = input_seq.size(0)
    #     output_seq = []
    #
    #     for t in range(seq_len):
    #         if output_seq:
    #             # Do NOT mutate previous output. Build new tensor combining old output and current input.
    #             prev_output = output_seq[-1]
    #             input_t = torch.cat([
    #                 input_seq[t],
    #                 prev_output[:, input_seq[t].size(1):]  # Remaining dimensions if any
    #             ], dim=1)
    #         else:
    #             input_t = input_seq[t]
    #
    #         output = self.sequential(input_seq[t])
    #         output_seq.append(output)
    #
    #         # print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    #
    #     output_seq = torch.stack(output_seq)
    #     output_seq = torch.transpose(output_seq, 0, 1)
    #     return output_seq


    # def forward(self, input_seq, batch_first=False):
    #
    #     if batch_first:
    #         input_seq = torch.transpose(input_seq, 0, 1)
    #
    #     seq_len = len(input_seq)
    #
    #     output_seq = []
    #     for t in range(seq_len):
    #         input_t = output_seq[-1] if output_seq else input_seq[t]
    #         input_t[:, :input_seq[t].size(1)] = input_seq[t]  # (batch_size, input_size)
    #
    #         output = self.sequential(input_t)  # (batch_size, output_size)
    #         output_seq.append(output)
    #
    #         print(f"Memory allocated: {torch.cuda.memory_allocated(torch.device('cuda')) / (1024 ** 2):.2f} MB")
    #
    #     return torch.stack(output_seq)

    # def forward(self, input_seq, hidden_0=None, batch_first=False):
    #     if batch_first:
    #         input_seq = input_seq.transpose(0, 1)  # Shape: (seq_len, batch_size, input_size)
    #
    #     seq_len, batch_size, input_size = input_seq.shape
    #     hidden_size = sum(self.sizes) - input_size
    #
    #     if hidden_0 is None:
    #         hidden_t = torch.zeros(batch_size, hidden_size, device=input_seq.device)
    #     else:
    #         hidden_t = hidden_0
    #
    #     outputs = []
    #
    #     for t in range(seq_len):
    #         input_t = input_seq[t]  # shape: (batch_size, input_size)
    #
    #         # Combine input and hidden state (do NOT use in-place ops!)
    #         combined = torch.cat((input_t, hidden_t[:, input_size:]), dim=1)  # shape: (batch_size, input_size + hidden_size)
    #
    #         hidden_t = self.sequential(combined)  # e.g., a linear + activation returning new hidden state
    #         outputs.append(hidden_t)
    #
    #         print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    #
    #     return torch.stack(outputs), hidden_t

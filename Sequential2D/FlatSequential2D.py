from Sequential2D import Sequential2D
import torch


class FlatSequential2D(torch.nn.Module):

    def __init__(
            self,
            block_in_features: list,
            block_out_features: list,
            blocks,
    ):
        super(FlatSequential2D, self).__init__()
        self.block_in_features = block_in_features
        self.block_out_features = block_out_features
        self.blocks = blocks

        self.sequential = Sequential2D(blocks)

    """
        Args:
            X: The input features of shape:
                (batch_size, in_features)
                  'batch_size'  - number of samples in the mini-batch
                  'in_features' - the total number of input features = sum(self.block_in_features_list)

        Returns:
            y: The output features of shape:
                (batch_size, out_features)
                  'batch_size'   - number of samples in the mini-batch
                  'out_features' - the total number of output features = sum(self.block_out_features_list)
    """
    def forward(self, X):
        in_blocks = [
            X[:, sum(self.block_in_features[:i]):sum(self.block_in_features[:i+1])]
            for i in range(len(self.block_in_features))
        ]

        out = torch.stack(self.sequential(in_blocks))  # (num_blocks, batch_size, block_size)
        out = torch.transpose(out, 0, 1)     # (batch_size, num_blocks, block_size)
        out = torch.flatten(out, start_dim=1)          # (batch_size, out_features)
        return out




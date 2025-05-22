import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence


def pad_and_pack_batch(batch_x, batch_y, device=None):
    # Get lengths
    x_lengths = torch.tensor([len(x) for x in batch_x], device=device)
    y_lengths = torch.tensor([len(x) for x in batch_y], device=device)

    # Sort by lengths (required for packing)
    x_lengths, obs_perm_idx = x_lengths.sort(descending=True)
    batch_x = [batch_x[i] for i in obs_perm_idx]

    y_lengths, act_perm_idx = y_lengths.sort(descending=True)
    batch_y = [batch_y[i] for i in act_perm_idx]

    # Pad sequences
    padded_x = pad_sequence(batch_x, batch_first=True)  # shape: (batch_size, max_len, features)
    padded_y = pad_sequence(batch_y, batch_first=True)

    # Pack sequences
    packed_x = pack_padded_sequence(padded_x, x_lengths, batch_first=True)
    packed_y = pack_padded_sequence(padded_y, y_lengths, batch_first=True)

    return packed_x, packed_y
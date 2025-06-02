import torch
import torch.nn as nn

def compute_loss(
        output,     # (batch_size, max_seq_len, 128) <- will take the last 128 of whatever size vector you give it
        batch_act,  # (batch_size, max_seq_len, 16)
        batch_sap   # (batch_size, max_seq_len, 32)
):
    output = output[:, :, -128:]  # (batch_size, max_seq_len, output_dim=128)
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    # Reshape targets
    flat_actions = batch_act.reshape(-1)         # (batch_size * max_seq_len * 16)
    flat_sap_deltas = batch_sap.reshape(-1, 32)  # (batch_size * max_seq_len, 32)

    # Reshape predictions
    actions_hat = output[:, :, :96]     # (batch_size, max_seq_len, 96)
    sap_deltas_hat = output[:, :, 96:]  # (batch_size, max_seq_len, 32)

    flat_actions_hat = torch.reshape(actions_hat, (-1, 6))         # (batch_size * max_seq_len * 16, 6)
    flat_sap_deltas_hat = torch.reshape(sap_deltas_hat, (-1, 32))  # (batch_size * max_seq_len, 32)

    ce_loss = ce_loss_fn(flat_actions_hat, flat_actions)
    mse_loss = mse_loss_fn(flat_sap_deltas_hat, flat_sap_deltas)

    return ce_loss, mse_loss

import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from lux.util.observation_to_tensor import observation_to_tensor


class LuxEpisodeDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations  # (n_samples, seq_len, input_dim)
        self.actions = actions            # (n_samples, seq_len, output_dim)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def action_collate_fn(batch):
    batch_obs, batch_act = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in batch_obs])

    # Sort by lengths (descending) for packing
    lengths, perm_idx = lengths.sort(0, descending=True)
    batch_obs = [batch_obs[i] for i in perm_idx]
    batch_act = [batch_act[i] for i in perm_idx]

    # Pad sequences
    padded_obs = pad_sequence(batch_obs, batch_first=True)  # (batch_size, max_seq_len, input_dim)
    padded_act = pad_sequence(batch_act, batch_first=True)  # (batch_size, max_seq_len, output_dim)

    return padded_obs, padded_act, lengths


def load_action_dataset(data_dir: str):
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    observations = []   # (n_samples, seq_len, in_size)
    actions = []        # (n_samples, seq_len, out_size)

    for file in files:
        try:
            with open(f'{data_dir}/{file}', 'r') as f:
                episode = json.load(f)
        except Exception as e:
            print(f'Failed to read {file}: {e}')
            continue

        obs, act = zip(*[observation_to_tensor(action_data) for action_data in episode])
        observations.append(torch.stack(obs))   # (seq_len, in_size)
        actions.append(torch.stack(act))        # (seq_len, out_size)

    return LuxEpisodeDataset(observations, actions)


if __name__ == "__main__":
    dataset = load_action_dataset('../data')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=action_collate_fn)

    for batch_obs, batch_act, lengths in dataloader:
        print(batch_obs.shape)
        print(batch_act.shape)


import os
import json
import torch
from torch.utils.data import Dataset
from lux.util.observation_to_tensor import observation_to_tensor


class LuxEpisodeDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def load_action_dataset(data_dir: str):
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    observations = []
    actions = []

    for file in files:
        try:
            with open(f'{data_dir}/{file}', 'r') as f:
                episode = json.load(f)
        except Exception as e:
            print(f'Failed to read {file}: {e}')
            continue

        obs, act = zip(*[observation_to_tensor(action_data) for action_data in episode])
        observations.append(obs)
        actions.append(act)

    return LuxEpisodeDataset(observations, actions)


if __name__ == "__main__":
    load_action_dataset('../data')

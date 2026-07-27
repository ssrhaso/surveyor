# Deployed as dino_wm/datasets/tworoom_dino_dset.py: h5-backed TwoRoom dataset
# for DINO-WM world-model training (the encoder-swap recovery leg). Mirrors
# PushTDataset's contract; pixels stay in the h5 (flat layout, ep_offset +
# ep_len boundaries) and the file handle opens lazily per worker process.
import pickle  # noqa: F401  (parity with sibling dsets)
from pathlib import Path
from typing import Callable, Optional

import hdf5plugin  # noqa: F401  (registers HDF5 compression filters)
import h5py
import numpy as np
import torch
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset


class TwoRoomDINODataset(TrajDataset):
    def __init__(
        self,
        h5_path: str,
        ep_indices,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
    ):
        self.h5_path = h5_path
        self.transform = transform
        self._f = None                      # opened lazily (fork-safe)
        with h5py.File(h5_path, "r") as f:
            off = np.asarray(f["ep_offset"][:])
            ln = np.asarray(f["ep_len"][:])
            act = torch.from_numpy(np.asarray(f["action"][:], dtype=np.float32))
            agent = torch.from_numpy(np.asarray(f["pos_agent"][:], dtype=np.float32))
            target = torch.from_numpy(np.asarray(f["pos_target"][:], dtype=np.float32))
        # each episode's final action row is NaN ("no action after last
        # state"); stats over valid rows, then zero-fill (poisoned epoch-1
        # loss to nan on the first training attempt, job 2299565)
        self._act_valid = ~torch.isnan(act).any(dim=-1)
        act = torch.nan_to_num(act, nan=0.0)
        self.ep_off = off[ep_indices]
        self.ep_len = ln[ep_indices]
        self.seq_lengths = [int(l) for l in self.ep_len]
        self.flat_actions = act
        self.flat_proprios = agent
        self.flat_states = torch.cat([agent, target], dim=-1)
        self.shapes = ["T"] * len(self.ep_off)

        self.action_dim = self.flat_actions.shape[-1]
        self.state_dim = self.flat_states.shape[-1]
        self.proprio_dim = self.flat_proprios.shape[-1]

        if normalize_action:
            self.action_mean = self.flat_actions[self._act_valid].mean(0)
            self.action_std = self.flat_actions[self._act_valid].std(0).clamp_min(1e-6)
            self.state_mean = self.flat_states.mean(0)
            self.state_std = self.flat_states.std(0).clamp_min(1e-6)
            self.proprio_mean = self.flat_proprios.mean(0)
            self.proprio_std = self.flat_proprios.std(0).clamp_min(1e-6)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
        self.flat_actions = (self.flat_actions - self.action_mean) / self.action_std
        self.flat_proprios = (self.flat_proprios - self.proprio_mean) / self.proprio_std
        print(f"Loaded {len(self.ep_off)} tworoom rollouts from {h5_path}")

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
        return self._f

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.ep_off)):
            o, l = int(self.ep_off[i]), int(self.ep_len[i])
            result.append(self.flat_actions[o:o + l])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        o = int(self.ep_off[idx])
        sel = [o + int(fr) for fr in frames]
        image = torch.from_numpy(self._file()["pixels"][sel]).float()  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        act = self.flat_actions[sel]
        state = self.flat_states[sel]
        proprio = self.flat_proprios[sel]
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {"shape": self.shapes[idx]}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.ep_off)


def load_tworoom_dino_slice_train_val(
    transform,
    data_path="/lustre/home/ha676/data/tworoom/tworoom.h5",
    n_rollout=None,
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    with h5py.File(data_path, "r") as f:
        n_total = len(f["ep_offset"])
    if n_rollout:
        n_total = min(n_total, n_rollout)
    n_train = int(n_total * split_ratio)
    train_dset = TwoRoomDINODataset(data_path, np.arange(0, n_train),
                                    transform=transform,
                                    normalize_action=normalize_action)
    val_dset = TwoRoomDINODataset(data_path, np.arange(n_train, n_total),
                                  transform=transform,
                                  normalize_action=normalize_action)
    num_frames = num_hist + num_pred
    datasets = {"train": TrajSlicerDataset(train_dset, num_frames, frameskip),
                "valid": TrajSlicerDataset(val_dset, num_frames, frameskip)}
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset

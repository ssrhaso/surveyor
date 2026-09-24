"""The authors' GC-IDM (arXiv 2605.08732, external/Latent-Geometry-...) served
through our GCIDMPolicy / SurveyorGCIDMPolicy contract.

Their `GoalConditionedIDM(z_t, z_goal, steps_remaining)` takes the raw number
of steps left (clamped at max_horizon, as their evaluator does) and divides by
max_horizon itself; it is trained on RAW actions (no scaler) from embeddings
that `train_idm.py extract` computes with the same encoder + projector path as
`encoder.encode_frames` (bit-identical, checked 2026-09-12). So the adapter
below only has to hand steps through and return raw actions.

`gcidm.load_gcidm` dispatches here when a checkpoint carries their keys
(`model_state_dict`, `config`), so every driver accepts
`--gcidm-ckpt gcidm_official_<env>_h<H>_s<seed>.pt` unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

OFFICIAL_DIR = (Path(__file__).resolve().parents[1] / "external"
                / "Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models")


def _idm():
    if str(OFFICIAL_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DIR))
    from idm.model import GoalConditionedIDM, IDMConfig
    return GoalConditionedIDM, IDMConfig


class OfficialGCIDM(nn.Module):
    """Adapter: exposes h_max, normalise_horizon and forward(z_t, z_goal, h)."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.h_max = int(model.max_horizon)
        self.official = True

    def normalise_horizon(self, steps_remaining):
        sr = torch.as_tensor(steps_remaining)
        return torch.clamp(sr, max=self.h_max).long()

    def forward(self, z_t, z_goal, h):
        return self.m(z_t, z_goal, h.to(z_t.device))


def load_official_gcidm(ck, device="cuda"):
    """`ck` is the torch.load'ed payload of train_idm.py; returns
    (adapter, action_scaler=None, meta) like gcidm.load_gcidm."""
    GoalConditionedIDM, IDMConfig = _idm()
    cfg = IDMConfig(**ck["config"])
    model = GoalConditionedIDM(cfg)
    model.load_state_dict(ck["model_state_dict"])
    adapter = OfficialGCIDM(model).to(device).eval()
    adapter.requires_grad_(False)
    meta = {"official": True, "epoch": ck.get("epoch"), "val_loss": ck.get("val_loss"),
            "held_out_episodes": len(ck.get("held_out_episodes", []) or []),
            "config": ck["config"]}
    return adapter, None, meta

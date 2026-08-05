"""V-JEPA 2-AC world-model wrapper: load, encode, frame-causal rollout.

Upstream imports are lazy so the rest of the package stays importable without
the clone's dependencies on the path (sources and drafter are pure torch).
Conventions mirror notebooks/utils/world_model_wrapper.py exactly: each frame is
duplicated along time to fill one tubelet before encoding, reps are layer-normed
under normalize_reps, and one predictor step appends the last tokens_per_frame
outputs as the next frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

UPSTREAM = Path(__file__).resolve().parent.parent / "upstream"


def ensure_upstream_on_path():
    """Make `src.*`, `app.*`, and the notebook `utils.*` importable."""
    for p in (str(UPSTREAM), str(UPSTREAM / "notebooks")):
        if p not in sys.path:
            sys.path.insert(0, p)


def pool(tokens: torch.Tensor) -> torch.Tensor:
    """Mean over the token axis: (..., T_tok, D) -> (..., D). The paper's
    instrument space for verification/c*/routing on this substrate."""
    return tokens.mean(dim=-2)


class VJEPA2WM:
    """Encoder + AC predictor + transform behind the four calls the method
    layer needs: encode_frames, step, rollout_plan, next_pose."""

    def __init__(self, model_name: str = "vjepa2_ac_vit_giant", device: str = "cuda",
                 crop_size: int = 256, normalize_reps: bool = True,
                 pretrained: bool = True):
        ensure_upstream_on_path()
        from app.vjepa_droid.transforms import make_transforms

        self.encoder, self.predictor = torch.hub.load(
            str(UPSTREAM), model_name, source="local", pretrained=pretrained)
        self.encoder.to(device).eval()
        self.predictor.to(device).eval()
        self.device = device
        self.normalize_reps = normalize_reps
        self.crop_size = crop_size
        self.tokens_per_frame = int((crop_size // self.encoder.patch_size) ** 2)
        self.embed_dim = int(self.encoder.embed_dim)
        self.transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=crop_size,
        )

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray, chunk: int = 32) -> torch.Tensor:
        """frames: (T, H, W, C) uint8 -> (T, tokens_per_frame, D) layer-normed
        reps, one tubelet per frame (frame duplicated to fill tubelet_size=2).
        Encoder forwards are chunked (`chunk` frames per pass) so long
        episodes fit in VRAM."""
        clip = self.transform(frames).unsqueeze(0)              # (1, C, T, H, W)
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        outs = []
        for s in range(0, T, chunk):
            part = clip[s:s + chunk].to(self.device, non_blocking=True)
            outs.append(self.encoder(part))
        h = torch.cat(outs, dim=0)                              # (T, tok, D)
        h = h.view(B, T, -1, h.size(-1))[0]
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    @torch.no_grad()
    def step(self, reps: torch.Tensor, actions: torch.Tensor,
             poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One frame-causal predictor step over the running context.
        reps (B, T, T_tok, D), actions (B, T, 7), poses (B, T, 7)
        -> next frame tokens (B, 1, T_tok, D), next pose (B, 1, 7)."""
        ensure_upstream_on_path()
        from utils.mpc_utils import compute_new_pose

        B, T, N_T, D = reps.size()
        reps = reps.to(self.device)
        actions = actions.to(self.device, dtype=reps.dtype)
        poses = poses.to(self.device, dtype=reps.dtype)
        next_rep = self.predictor(reps.flatten(1, 2), actions, poses)[:, -self.tokens_per_frame:]
        if self.normalize_reps:
            next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
        next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
        return next_rep.view(B, 1, N_T, D), next_pose

    @torch.no_grad()
    def rollout_plan(self, z0: torch.Tensor, s0: torch.Tensor,
                     action_traj: torch.Tensor) -> torch.Tensor:
        """Roll a fixed action trajectory from one start state.
        z0 (B, T_tok, D), s0 (B, 1, 7), action_traj (B, H, 7)
        -> terminal frame tokens (B, T_tok, D)."""
        reps, poses = z0.unsqueeze(1), s0
        acts = None
        for h in range(action_traj.shape[1]):
            a = action_traj[:, h:h + 1]
            acts = a if acts is None else torch.cat([acts, a], dim=1)
            nxt, npse = self.step(reps, acts, poses)
            reps = torch.cat([reps, nxt], dim=1)
            poses = torch.cat([poses, npse], dim=1)
        return reps[:, -1]

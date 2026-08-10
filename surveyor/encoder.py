"""Shared I/O for the frozen LeWM world model: loading, frame encoding, the
canonical PushT target, and tolerance-parameterized success criteria.

The pretrained and local load paths produce the identical frozen model. Nothing
here is trained or unfrozen.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

# Canonical PushT target (the green T), fixed across the dataset because the
# `goal` variation is NOT in DEFAULT_VARIATIONS, so reset() never randomizes it.
# Verified against env.goal_pose == [256, 256, pi/4]. State layout (7-d), from
# PushT._get_obs: [agent_x, agent_y, block_x, block_y, block_angle, vx, vy].
TARGET_BLOCK_XY = np.array([256.0, 256.0])
TARGET_BLOCK_ANGLE = float(np.pi / 4)
POS_THRESH = 20.0  # env-native positional tolerance (pixels, 4-D agent+block)


# TwoRoom (stable_worldmodel/envs/two_room/env.py). Unlike PushT the target is
# sampled per episode and recorded in the h5 (state=agent xy, goal_state=target
# xy, distance_to_target=precomputed dist), so nothing needs reconstructing.
# Matches env.py step(): terminated = dist(agent, target) < 16.0, no angle.
TWOROOM_POS_THRESH = 16.0


def tworoom_success(dist_to_target: float, pos_thresh: float = TWOROOM_POS_THRESH) -> bool:
    """TwoRoomEnv.step's success rule, applied to a recorded distance_to_target."""
    return bool(float(dist_to_target) < pos_thresh)


def eval_state_tol(goal_state, cur_state, angle_deg, pos_thresh: float = POS_THRESH):
    """Tolerance-parameterized copy of PushT.eval_state, verified identical to
    the bound env method at 20 deg. Returns the success bool only."""
    goal_state = np.asarray(goal_state, dtype=np.float64)
    cur_state = np.asarray(cur_state, dtype=np.float64)
    pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
    angle_diff = np.abs(goal_state[4] - cur_state[4])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    return bool(pos_diff < pos_thresh and angle_diff < np.radians(angle_deg))


def canonical_target_for(final_state):
    """Build the 7-D canonical goal_state for an episode's final state.

    The block pose becomes the canonical target and the agent terms are copied
    from final_state so they contribute zero, which gives the block-only success
    criterion under the exact env math."""
    final_state = np.asarray(final_state, dtype=np.float64)
    target = final_state.copy()
    target[2:4] = TARGET_BLOCK_XY
    target[4] = TARGET_BLOCK_ANGLE
    return target


# Model loading (frozen)
def _ensure_swm_importable(swm_src: str | None):
    """Make `stable_worldmodel` importable without a full install (CPU path) by
    adding the source checkout to sys.path and stubbing the unused lancedb dep."""
    try:
        import stable_worldmodel  # noqa: F401
        return
    except Exception:
        pass
    if "lancedb" not in sys.modules:
        m = types.ModuleType("lancedb")
        perm = types.ModuleType("lancedb.permutation")
        perm.Permutation = object
        m.permutation = perm
        m.connect = lambda *a, **k: None
        sys.modules["lancedb"] = m
        sys.modules["lancedb.permutation"] = perm
    if swm_src and str(swm_src) not in sys.path:
        sys.path.insert(0, str(swm_src))


def _build_vit():
    """Replicate stable_pretraining.backbone.utils.vit_hf('tiny', patch=14, 224)."""
    from transformers import ViTConfig, ViTModel

    cfg = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=192 * 4,
        image_size=224,
        patch_size=14,
    )
    model = ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)
    model.config.interpolate_pos_encoding = True
    return model


def _load_local(local_dir: str):
    """Reconstruct LeWM from a dir with config.json + weights.pt (no hydra)."""
    from stable_worldmodel.wm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import Predictor, Embedder, MLP

    local_dir = Path(local_dir)
    cfg = json.loads((local_dir / "config.json").read_text())
    enc = _build_vit()

    pc = cfg["predictor"]
    predictor = Predictor(
        num_frames=pc["num_frames"], input_dim=pc["input_dim"],
        hidden_dim=pc["hidden_dim"], output_dim=pc["output_dim"],
        depth=pc["depth"], heads=pc["heads"], mlp_dim=pc["mlp_dim"],
        dim_head=pc["dim_head"], dropout=pc["dropout"], emb_dropout=pc["emb_dropout"],
    )
    ac = cfg["action_encoder"]
    action_encoder = Embedder(input_dim=ac["input_dim"], emb_dim=ac["emb_dim"])

    def mk_mlp(d):
        return MLP(input_dim=d["input_dim"], hidden_dim=d["hidden_dim"],
                   output_dim=d["output_dim"], norm_fn=torch.nn.BatchNorm1d)

    model = LeWM(
        encoder=enc, predictor=predictor, action_encoder=action_encoder,
        projector=mk_mlp(cfg["projector"]), pred_proj=mk_mlp(cfg["pred_proj"]),
    )
    sd = torch.load(local_dir / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    return model


def load_lewm(source="pretrained", encoder_id="quentinll/lewm-pusht",
              local_dir=None, swm_src=None, device="cpu"):
    """Load and FREEZE the LeWM world model. Returns model in eval() mode with
    requires_grad_(False). `source` in {"pretrained","local"}."""
    if source == "pretrained":
        _ensure_swm_importable(swm_src)  # no-op if pip-installed
        from stable_worldmodel.wm.utils import load_pretrained
        model = load_pretrained(encoder_id)
    elif source == "local":
        if local_dir is None:
            raise ValueError("source='local' requires local_dir")
        _ensure_swm_importable(swm_src)
        model = _load_local(local_dir)
    else:
        raise ValueError(f"unknown source {source!r}")

    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


# Pixel preprocessing + encoding (matches le-wm get_img_preprocessor / harness:
# uint8 -> /255 -> ImageNet normalize; env renders at 224 so no resize needed)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess_frames(frames):
    """frames: (N,H,W,3) uint8 -> (N,3,224,224) float32, ImageNet-normalized."""
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    x = frames.permute(0, 3, 1, 2).float() / 255.0
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


@torch.no_grad()
def encode_frames(model, frames, device="cpu", batch_size=256):
    """Encode frames (N,H,W,3) uint8 to (N,192) latents via the exact LeWM.encode
    path. eval() is required so projector BatchNorm uses running stats, keeping
    latents independent of batch composition."""
    assert not model.training, "model must be in eval() before encoding"
    x = preprocess_frames(frames)
    out = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device)
        info = {"pixels": xb.unsqueeze(1)}  # (b,1,3,224,224)
        emb = model.encode(info)["emb"][:, 0]  # (b,192)
        out.append(emb.float().cpu())
    return torch.cat(out, 0)

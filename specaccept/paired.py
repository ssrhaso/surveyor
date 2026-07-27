"""Paired-latent spec-accept: plan in the world model's space, CERTIFY in another.

Motivation (TwoRoom, measured). LeWM's TwoRoom encoder is metrically
degenerate: consecutive frames sit at rel L2 p50 0.876 while unrelated frames
sit at 1.456, and equiv p90 (1.393) exceeds hop p10 (1.018), so no threshold
separates "arrived" from "not arrived" (Results/gap_stat/gap_tworoom*.json).
Flat CEM planning is untouched by this -- it never consults that metric -- and
reaches 68 percent at t=25. Only the VERIFIER is blocked.

The fix is to decouple the two roles. The planner keeps consuming LeWM
latents; the verifier certifies arrival in a space where the gap is open
(frozen DINOv2). Both halves of a waypoint have to describe the SAME imagined
future, so a single drafter emits the concatenation
    z = [ z_lewm (192) | z_dino (384) ]
and we split it at serving: the LeWM half goes to SubgoalCostModel, the DINOv2
half is what the accept test compares. Drafting them independently would let
the two halves disagree about which future they mean.

This generalizes the lens (learn_readout), which already certifies in a learned
readout of the planner's space rather than the space itself; here the readout
is simply a frozen general-purpose encoder instead of a trained one.
"""

from __future__ import annotations

import numpy as np
import torch

from specaccept import encoder

LEWM_DIM = 192
DINO_DIM = 384


# ---------------------------------------------------------------- DINOv2 side
def load_dinov2(device="cpu", name="dinov2_vits14"):
    """Frozen DINOv2 ViT-S/14, the same hub model the DINO-WM leg and the
    tworoom gap probes use (TORCH_HOME must point at the shared cache)."""
    model = torch.hub.load("facebookresearch/dinov2", name)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode_frames_dino(model, frames, device="cpu", batch_size=128):
    """frames (N,H,W,3) uint8 -> (N,384) pooled patch tokens.

    Preprocessing is encoder.preprocess_frames verbatim (uint8 -> /255 ->
    ImageNet normalize; TwoRoom renders natively at 224 so there is no resize),
    and pooling is the mean over patch tokens -- the space the gap statistic is
    computed in. Any tau used with this function must be derived from a probe
    that encodes through this same function."""
    assert not model.training, "DINOv2 must be in eval() before encoding"
    x = encoder.preprocess_frames(frames)
    out = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device)
        z = model.forward_features(xb)["x_norm_patchtokens"]   # (b, P, 384)
        out.append(z.mean(dim=1).float().cpu())
    return torch.cat(out, 0)


class PairedEncoder:
    """Encodes frames into [lewm | dino] once, so the builder and the serving
    path cannot drift apart."""

    def __init__(self, lewm, dino, device="cpu"):
        self.lewm = lewm
        self.dino = dino
        self.device = device

    @torch.no_grad()
    def encode(self, frames, batch_size=128):
        zl = encoder.encode_frames(self.lewm, frames, device=self.device,
                                   batch_size=batch_size)
        zd = encode_frames_dino(self.dino, frames, device=self.device,
                                batch_size=batch_size)
        return torch.cat([zl, zd], dim=-1)


def split_paired(z):
    """(..., 576) -> ((..., 192) lewm, (..., 384) dino)."""
    return z[..., :LEWM_DIM], z[..., LEWM_DIM:]


# ------------------------------------------------------------------- source
class SpecAcceptPairedSource:
    """Spec-accept whose verifier lives in the DINOv2 half of a paired draft.

    Identical control flow to SpecAcceptSubgoalSource (draft a block of N
    waypoints; at each replan accept the pursued waypoint iff the achieved
    state verifies against it, else re-draft from reality), with two changes:
      * the drafted block is (N, 576); the LeWM half is served to the cost
        model and the DINOv2 half is what the accept test compares;
      * the achieved state is encoded by BOTH encoders, so the source needs the
        raw frames at each replan (wants_frames).

    tau is a relative L2 in the pooled-DINOv2 space and MUST come from a gap
    probe run through encode_frames_dino at the serving hop -- never
    transferred from another space.
    """

    needs_obs = True
    wants_frames = True

    def __init__(self, planner, paired_encoder, n_envs, device="cpu", n_steps=50,
                 seed=42, tau=0.2, record=False, readout=None, readout_tau=None):
        # readout: optional time-contrastive lens over the DINOv2 half. The
        # TwoRoom gap in raw pooled DINOv2 is CLOSED at every hop (equiv p90
        # 0.210 vs hop10 p10 0.074) with bulks separated ~1.7x -- the reacher
        # shape, for which the instrument prescribes a lens rather than a fixed
        # threshold. When set, accept iff ||r(d_now) - r(d_tgt)|| <= readout_tau.
        self.readout = readout
        self.readout_tau = None if readout_tau is None else float(readout_tau)
        self.planner = planner
        self.penc = paired_encoder
        self.needs_goal = bool(getattr(planner, "goal_cond", False))
        self.N = int(planner.cfg.n_future)
        self.dim = int(planner.cfg.latent_dim)
        assert self.dim == LEWM_DIM + DINO_DIM, (
            f"paired drafter must be {LEWM_DIM + DINO_DIM}-d, got {self.dim}; "
            "rebuild subgoals with --dino-pair and retrain")
        self.device = device
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.tau = float(tau)
        self._queue = [None] * n_envs            # (N, 576) drafted block
        self._ptr = np.zeros(n_envs, dtype=int)
        self._target = [None] * n_envs           # (384,) DINOv2 half being pursued
        self._cache = torch.zeros(n_envs, LEWM_DIM, device=device)
        self._gen = torch.Generator(device=planner.device)
        self._gen.manual_seed(int(seed))
        self.record = record
        self.trace = []
        self.n_redraft = 0
        self.n_advance = 0
        self.n_reject = 0
        self.rels = []       # every verification distance, for the mechanics report

    @torch.no_grad()
    def _pair(self, frames, lewm_latent=None):
        """[lewm | dino] for these frames, reusing the policy's LeWM encode."""
        zl = (lewm_latent.to(self.device) if lewm_latent is not None
              else encoder.encode_frames(self.penc.lewm, frames,
                                         device=self.penc.device).to(self.device))
        zd = encode_frames_dino(self.penc.dino, frames,
                                device=self.penc.device).to(self.device)
        return torch.cat([zl, zd], dim=-1)

    @torch.no_grad()
    def current(self, sg_steps, obs_latent=None, replan_idx=None, goal_latent=None,
                frames=None, goal_frames=None) -> torch.Tensor:
        if replan_idx is None or len(replan_idx) == 0 or frames is None:
            return self._cache.clone()

        # achieved paired latent for the replanning envs. The policy has
        # already run the LeWM encode (obs_latent), so only the DINOv2 half is
        # computed here; both halves come from the SAME frames either way.
        z_now = self._pair(frames, obs_latent)                        # (R, 576)
        _, d_now = split_paired(z_now)
        z_goal_paired = None
        if self.needs_goal and goal_frames is not None:
            z_goal_paired = self._pair(goal_frames, goal_latent)

        need_rows, need_envs = [], []
        for r, i in enumerate(replan_idx):
            q, tgt = self._queue[i], self._target[i]
            accept = False
            if q is not None and tgt is not None:
                if self.readout is not None:
                    rel = float((self.readout(d_now[r]) - self.readout(tgt)).norm())
                    verified = rel <= self.readout_tau
                else:
                    rel = float((d_now[r] - tgt).norm() / tgt.norm().clamp_min(1e-8))
                    verified = rel <= self.tau
                self.rels.append(rel)
                if not verified:
                    self.n_reject += 1
                accept = verified and self._ptr[i] < self.N
            if accept:
                nxt = q[self._ptr[i]]
                self._ptr[i] += 1
                self._target[i] = split_paired(nxt)[1]
                self._cache[i] = split_paired(nxt)[0]
                self.n_advance += 1
            else:
                need_rows.append(r)
                need_envs.append(i)

        if need_rows:
            z_cond = z_now[need_rows].to(self.planner.device)
            z_goal = (z_goal_paired[need_rows].to(self.planner.device)
                      if z_goal_paired is not None else None)
            blocks = self.planner.sample_sequence(z_cond, n_steps=self.n_steps,
                                                  generator=self._gen,
                                                  z_goal_native=z_goal)   # (R', N, 576)
            blocks = blocks.to(self.device)
            if self.record:
                self.trace.append({"replan_idx": np.asarray(need_envs),
                                   "z_cond": z_cond.detach().float().cpu(),
                                   "block": blocks.detach().float().cpu()})
            for j, i in enumerate(need_envs):
                self._queue[i] = blocks[j]
                self._ptr[i] = 1
                self.n_redraft += 1
                self._target[i] = split_paired(blocks[j][0])[1]
                self._cache[i] = split_paired(blocks[j][0])[0]
        return self._cache.clone()

    def stats(self, tau=None):
        total = self.n_redraft + self.n_advance
        rels = np.asarray(self.rels) if self.rels else np.array([0.0])
        space = "dino384-lens" if self.readout is not None else "dino384"
        tau_used = self.readout_tau if self.readout is not None else self.tau
        return (f"[specpaired] tau={tau_used} verify-space={space} "
                f"re-drafts={self.n_redraft} advances={self.n_advance} "
                f"rejects={self.n_reject} "
                f"call_ratio={self.n_redraft / max(total, 1):.3f} "
                f"(every-step=1.000) | verification distance "
                f"p10/p50/p90={np.percentile(rels, 10):.3f}/"
                f"{np.percentile(rels, 50):.3f}/{np.percentile(rels, 90):.3f}")

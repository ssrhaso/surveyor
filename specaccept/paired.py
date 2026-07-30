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
                 seed=42, tau=0.2, record=False, readout=None, readout_tau=None,
                 goal_gate=False, snap_bank=None, snap_progress=False,
                 best_of_k=1, bok_score="goal", route_goal_hop=None,
                 verify_half="dino"):
        # Best-of-k drafting. Sample k candidate blocks per re-draft and serve
        # the one that scores best IN THE DINOv2 HALF -- the same principle as
        # the two mechanisms that measured positive on TwoRoom (verify in
        # DINOv2, jointly-trained halves): use the structured space for
        # DECISIONS, the planner's space for execution. Scoring rules, both
        # zero-constant reuses of the verifier's own metric:
        #   goal  rel L2 of the block's FINAL waypoint to the goal (progress);
        #   feas  rel L2 of the block's FIRST waypoint to the current state
        #         (achievability -- the quantity verification will test).
        # k is derived OFFLINE by probe_bok.py (saturation of the selected
        # score), never against a closed-loop number.
        self.best_of_k = int(best_of_k)
        self.bok_score = str(bok_score)
        assert self.bok_score in ("goal", "feas")
        self.bok_margins = []   # median-candidate minus selected score, per pick
        # Proximity router (short-horizon protection). At each replan, if the
        # GOAL itself is within one certified hop in the verification space
        # (rel L2 <= route_goal_hop, the measured hop-S10 scale from the gap
        # probe -- a derived constant, not tuned), serve the goal directly and
        # draft nothing: an intermediate waypoint cannot help inside one
        # serving stride, and the oracle measured decomposition NEGATIVE at
        # short horizon. This is the c*-retire idea transplanted to the
        # verification space; unlike the (null) arrival gate it is re-evaluated
        # EVERY replan, not a one-way retirement.
        self.route_goal_hop = None if route_goal_hop is None else float(route_goal_hop)
        self.n_route = 0
        # Which half the ACCEPT TEST reads. 'dino' is the method (certify in
        # the external structured space); 'lewm' is the control arm for the
        # verify-space question on substrates whose own gap is open (pusht
        # C1/C2): same drafter, same serving, only the verifier's ruler
        # changes. Router/bok/snap stay in the DINOv2 half regardless (their
        # constants were derived there).
        self.verify_half = str(verify_half)
        assert self.verify_half in ("dino", "lewm")
        # readout: optional time-contrastive lens over the DINOv2 half. The
        # TwoRoom gap in raw pooled DINOv2 is CLOSED at every hop (equiv p90
        # 0.210 vs hop10 p10 0.074) with bulks separated ~1.7x -- the reacher
        # shape, for which the instrument prescribes a lens rather than a fixed
        # threshold. When set, accept iff ||r(d_now) - r(d_tgt)|| <= readout_tau.
        self.readout = readout
        self.readout_tau = None if readout_tau is None else float(readout_tau)
        # Arrival gate. Once the achieved state verifies against the FINAL GOAL
        # the env retires, one way, and the goal is served for the rest of the
        # episode at zero drafter cost. This is the mechanism that turned Cube
        # from a loss into a win (+32pp of the total margin there came from the
        # overshoot tax alone). It reuses the accept test, so it adds no tuned
        # constant. TwoRoom gives twice the budget the goal needs, so there is
        # ample room to arrive and then be walked back off the target.
        self.goal_gate = bool(goal_gate)
        self.retired = np.zeros(n_envs, dtype=bool)
        self.n_gate = 0
        # Retrieval snapping. LeWM's TwoRoom encoder puts consecutive frames at
        # rel L2 0.876 against a random-pair baseline of 1.456, so a GENERATED
        # latent in that space need not correspond to any reachable state, and
        # the closed loop confirms it: achieved-vs-waypoint distance sits at
        # 0.211 against a cross-pair baseline of 0.226. Flat is unharmed because
        # its target is a real encoded frame. So snap every drafted waypoint to
        # its nearest REAL frame from the training bank, making waypoints
        # reachable by construction. The nearest-neighbour search runs in the
        # DINOv2 half, which is well structured (equiv 0.098 vs cross 0.226),
        # and the retrieved frame's LeWM half is what the planner is served, so
        # both halves come from one real frame and cannot disagree.
        self.snap = None
        if snap_bank is not None:
            bank = snap_bank.to(device).float()
            self.snap = bank
            self.snap_d = split_paired(bank)[1].contiguous()
            self.snap_dn = self.snap_d / self.snap_d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.snap_progress = bool(snap_progress)
        self.n_snap = 0
        self.n_prog_empty = 0
        self.snap_moves = []
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
        # per-replan event log for filmstrips: exactly ONE (env, kind, rel)
        # appended per env per current() call, so the k-th event of env i
        # aligns with the k-th strip frame the policy captured for env i.
        # kind: 'advance' (verified, waypoint served from queue), 'redraft'
        # (rejected or no queue), 'gate' (retired, goal served).
        self.events = []

    @torch.no_grad()
    def _snap_block(self, block, d_now=None, d_goal=None):
        """(R, N, 576) drafted -> (R, N, 576) with each waypoint replaced by a
        REAL training frame, matched on the DINOv2 half by cosine.

        With snap_progress and the current/goal latents supplied, candidates are
        first restricted to bank frames strictly CLOSER to the goal than the
        agent already is. A waypoint that does not reduce distance to the goal
        is not a subgoal, and unconstrained nearest-neighbour will happily
        return one, since the drafted point it is matching may itself point
        nowhere useful. Rows with no qualifying candidate fall back to the
        unconstrained match rather than being dropped."""
        if self.snap is None:
            return block
        R, N, _ = block.shape
        q = split_paired(block.reshape(R * N, -1))[1]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sim = qn @ self.snap_dn.T                           # (R*N, B)

        if self.snap_progress and d_now is not None and d_goal is not None:
            gn = d_goal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            d_bank_goal = torch.cdist(d_goal, self.snap_d) / gn      # (R, B)
            d_now_goal = ((d_now - d_goal).norm(dim=-1, keepdim=True) / gn)  # (R, 1)
            ok = d_bank_goal < d_now_goal                            # (R, B)
            self.n_prog_empty += int((~ok.any(dim=-1)).sum())
            ok = ok.repeat_interleave(N, dim=0)                      # (R*N, B)
            keep = ok.any(dim=-1, keepdim=True)
            sim = torch.where(ok | ~keep, sim, torch.full_like(sim, -2.0))

        idx = sim.argmax(dim=-1)                            # (R*N,)
        snapped = self.snap[idx]
        # how far each draft had to move, in the verification metric: near zero
        # means the drafter was already on-manifold and snapping is a no-op
        moved = ((q - split_paired(snapped)[1]).norm(dim=-1)
                 / split_paired(snapped)[1].norm(dim=-1).clamp_min(1e-8))
        self.snap_moves.extend(moved.tolist())
        self.n_snap += R * N
        return snapped.reshape(R, N, -1)

    @torch.no_grad()
    def _accepts(self, d_a, d_b, want_dist=False):
        """Is d_a at d_b, in the verification space? One test, used for both
        waypoint acceptance and the arrival gate, so the gate stays free of
        any threshold of its own."""
        if self.readout is not None:
            dist = float((self.readout(d_a) - self.readout(d_b)).norm())
            ok = dist <= self.readout_tau
        else:
            dist = float((d_a - d_b).norm() / d_b.norm().clamp_min(1e-8))
            ok = dist <= self.tau
        return (ok, dist) if want_dist else ok

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
        # the half the ACCEPT TEST reads (router/bok/snap stay on d_now/dino)
        v_now = d_now if self.verify_half == "dino" else split_paired(z_now)[0]
        z_goal_paired = None
        if self.needs_goal and goal_frames is not None:
            z_goal_paired = self._pair(goal_frames, goal_latent)

        # arrival gate: retire envs that have verifiably reached the final goal
        if self.goal_gate and z_goal_paired is not None:
            v_goal = (split_paired(z_goal_paired)[1] if self.verify_half == "dino"
                      else split_paired(z_goal_paired)[0])
            for r, i in enumerate(replan_idx):
                if self.retired[i]:
                    continue
                if self._accepts(v_now[r], v_goal[r]):
                    self.retired[i] = True

        need_rows, need_envs = [], []
        for r, i in enumerate(replan_idx):
            if self.retired[i]:                      # hold the goal, no drafting
                self._cache[i] = split_paired(z_goal_paired[r])[0]
                self.n_gate += 1
                self.events.append((int(i), "gate", float("nan")))
                continue
            if self.route_goal_hop is not None and z_goal_paired is not None:
                d_goal_r = split_paired(z_goal_paired[r])[1]
                rel_g = float((d_now[r] - d_goal_r).norm()
                              / d_goal_r.norm().clamp_min(1e-8))
                if rel_g <= self.route_goal_hop:     # goal within one hop: no
                    self._cache[i] = split_paired(z_goal_paired[r])[0]  # drafting
                    self.n_route += 1
                    self.events.append((int(i), "route", rel_g))
                    continue
            q, tgt = self._queue[i], self._target[i]
            accept = False
            rel = float("nan")
            if q is not None and tgt is not None:
                verified, rel = self._accepts(v_now[r], tgt, want_dist=True)
                self.rels.append(rel)
                if not verified:
                    self.n_reject += 1
                accept = verified and self._ptr[i] < self.N
            if accept:
                nxt = q[self._ptr[i]]
                self._ptr[i] += 1
                self._target[i] = (split_paired(nxt)[1] if self.verify_half == "dino"
                                   else split_paired(nxt)[0])
                self._cache[i] = split_paired(nxt)[0]
                self.n_advance += 1
                self.events.append((int(i), "advance", float(rel)))
            else:
                need_rows.append(r)
                need_envs.append(i)
                self.events.append((int(i), "redraft", float(rel)))

        if need_rows:
            z_cond = z_now[need_rows].to(self.planner.device)
            z_goal = (z_goal_paired[need_rows].to(self.planner.device)
                      if z_goal_paired is not None else None)
            K = self.best_of_k
            if K > 1:
                if self.bok_score == "goal" and z_goal is None:
                    raise ValueError("--bok-score goal needs a goal-conditioned "
                                     "stack (no goal frames reached the source)")
                Rn = z_cond.shape[0]
                blocks = self.planner.sample_sequence(
                    z_cond.repeat_interleave(K, dim=0), n_steps=self.n_steps,
                    generator=self._gen,
                    z_goal_native=(z_goal.repeat_interleave(K, dim=0)
                                   if z_goal is not None else None))
                blocks = blocks.reshape(Rn, K, self.N, -1)     # (R', K, N, 576)
                d_cand = split_paired(blocks)[1]               # (R', K, N, 384)
                if self.bok_score == "goal":
                    ref = split_paired(z_goal)[1]              # (R', 384)
                    pt = d_cand[:, :, -1]                      # final waypoint
                else:
                    ref = d_now[need_rows].to(blocks.device)   # (R', 384)
                    pt = d_cand[:, :, 0]                       # first waypoint
                s = ((pt - ref[:, None]).norm(dim=-1)
                     / ref.norm(dim=-1, keepdim=True).clamp_min(1e-8))  # (R', K)
                pick = s.argmin(dim=1)
                self.bok_margins.extend(
                    (s.median(dim=1).values - s.gather(1, pick[:, None])[:, 0])
                    .tolist())
                blocks = blocks[torch.arange(Rn, device=blocks.device), pick]
            else:
                blocks = self.planner.sample_sequence(z_cond, n_steps=self.n_steps,
                                                      generator=self._gen,
                                                      z_goal_native=z_goal)   # (R', N, 576)
            g_rows = (split_paired(z_goal_paired[need_rows])[1]
                      if z_goal_paired is not None else None)
            blocks = self._snap_block(blocks.to(self.device),
                                      d_now=d_now[need_rows], d_goal=g_rows)
            if self.record:
                self.trace.append({"replan_idx": np.asarray(need_envs),
                                   "z_cond": z_cond.detach().float().cpu(),
                                   "block": blocks.detach().float().cpu()})
            for j, i in enumerate(need_envs):
                self._queue[i] = blocks[j]
                self._ptr[i] = 1
                self.n_redraft += 1
                self._target[i] = (split_paired(blocks[j][0])[1]
                                   if self.verify_half == "dino"
                                   else split_paired(blocks[j][0])[0])
                self._cache[i] = split_paired(blocks[j][0])[0]
        return self._cache.clone()

    def stats(self, tau=None):
        total = self.n_redraft + self.n_advance
        rels = np.asarray(self.rels) if self.rels else np.array([0.0])
        space = ("dino384-lens" if self.readout is not None
                 else "dino384" if self.verify_half == "dino" else "lewm192")
        tau_used = self.readout_tau if self.readout is not None else self.tau
        gate = (f" gated_serves={self.n_gate} retired={int(self.retired.sum())}/"
                f"{self.n_envs}" if self.goal_gate else "")
        if self.snap is not None and self.snap_moves:
            mv = np.asarray(self.snap_moves)
            gate += (f" snapped={self.n_snap} bank={self.snap.shape[0]} "
                     f"move_p50={np.percentile(mv, 50):.3f}")
            if self.snap_progress:
                gate += f" prog_empty={self.n_prog_empty}"
        if self.best_of_k > 1 and self.bok_margins:
            bm = np.asarray(self.bok_margins)
            gate += (f" bok={self.best_of_k}({self.bok_score}) "
                     f"sel_margin p50={np.percentile(bm, 50):.3f} "
                     f"p90={np.percentile(bm, 90):.3f}")
        if self.route_goal_hop is not None:
            gate += (f" router_hop={self.route_goal_hop} "
                     f"routed={self.n_route}")
        return (f"[specpaired]{gate} tau={tau_used} verify-space={space} "
                f"re-drafts={self.n_redraft} advances={self.n_advance} "
                f"rejects={self.n_reject} "
                f"call_ratio={self.n_redraft / max(total, 1):.3f} "
                f"(every-step=1.000) | verification distance "
                f"p10/p50/p90={np.percentile(rels, 10):.3f}/"
                f"{np.percentile(rels, 50):.3f}/{np.percentile(rels, 90):.3f}")

# V-JEPA 2 transplant: certified spec-accept at video scale

The ICLR generality result (PLAN.md "NEXT"): port the paper's method (spec-accept
drafting plus the c* arbiter, one rule, one tau=0.20) from the 18M LeWM stack onto
V-JEPA 2's video-scale world model. The efficiency section of `paper/main_v2.tex`
already stakes the claim this experiment cashes in: in the LeWM stack the drafter is
0.4-1.0% of wall-clock (CEM dominates), but the drafter tier is where model capacity
grows; on a video-scale drafter the same consumption policy inherits the full NFE
reduction *as wall-clock*. This folder is where that gets tested.

## Layout

- `upstream/`: shallow clone of https://github.com/facebookresearch/vjepa2,
  pinned at `204698b45b37` (2026-03-23, "Fix figure (#143)"). Untracked scratch;
  re-clone or `git -C upstream fetch --depth 1` to bump, and update the pin here.
  (Windows note: the clone warns about case-colliding `vitG-384`/`vitg-384` config
  yamls; harmless, code paths unaffected.)
- `specaccept_vjepa2/`: **the transplant, implemented** (2026-07-20). Imports
  upstream lazily and `specaccept.drafter`'s validated diffusion machinery
  directly; does not edit upstream in place.
- `run_probe_isca.sbatch`: first GPU run (offline probe) for ISCA.

## The implemented package (`specaccept_vjepa2/`)

Space contract (the one transplant decision, stated in every docstring):
targets handed to the planner are **token grids** passed as `goal_frame` to
upstream's *unmodified* `cem()`; serving the true goal tokens is bit-identical
to Meta's flat protocol (the paper's reduction property, preserved exactly).
Verification, c*, retirement, and routing read **mean-pooled** vectors with
relative L2, the paper's instrument; tau=0.20 is the frozen transfer hypothesis.

| module | contents |
|---|---|
| `wm.py` | `VJEPA2WM`: hub-local load of encoder + AC predictor, frame-to-token encode (notebook-exact: tubelet duplication + layer-norm), frame-causal `step`/`rollout_plan`; `pool()` |
| `planner.py` | `flat_plan()` = pass-through to upstream `cem()` with arbitrary target tokens; `cstar()` = port of `cem_flat_cstar` (solve flat, roll the plan, rel pooled terminal distance) |
| `drafter.py` | `TokenGDM`: DiT denoiser over N x 256-token subgoal blocks, factorized block/token positions, pooled conditioning (+goal); keeps `GDM`'s forward contract so `specaccept.drafter.GaussianDiffusion` is reused **verbatim**; `TokenGDMPlanner`, save/load |
| `sources.py` | `SpecAcceptTokenSource` (tau-verify, serve-from-queue, redraft; optional cube-style goal-gate) and `CstarRetireTokenSource` (per-replan c*, one-way retire, first-replan fire test = router), line-faithful ports with the same stats fields; `LerpBlockDrafter` (data-free decomposition control), `GDMDraft` adapter |
| `policy.py` | `CertifiedSpecAcceptPolicy`: upstream's deploy loop verbatim, with the CEM target swapped by the certified source; `use_certificate=False` + lerp `frac=1.0` reduces to flat (anchor arm) |
| `train_drafter.py` | pairs builder (S-strided token-grid blocks, pooled cond/goal) + training loop (cosine/v-param/Min-SNR, the LeWM-stack recipe) |
| `probe_offline.py` | the stage-0 runnable: ANCHOR (notebook CEM sanity), C* ladder vs offset (with saturating raw-distance contrast), criterion-FLOOR shape (is tau=0.20 sane here?), PLUMBING (teacher-forced certified source along the recorded episode, lerp drafter); all on the bundled Franka traj, no robot/sim |
| `smoke_test.py` | CPU, tiny tensors, no weights: **all passing** (drafter train+sample, verify/accept/reject/redraft, retire+router fire test, adapters, pairs) |

Run order:
1. `python -m specaccept_vjepa2.smoke_test`: passes on CPU already.
2. `probe_offline.py` (GPU, or CPU with default tiny CEM budgets; first run
   downloads ViT-g AC weights, several GB) decides whether tau transfers and
   whether c* separates on this substrate. `run_probe_isca.sbatch` is the ISCA job.
3. `--dump-latents` output feeds `train_drafter.py` for an overfit smoke; a real
   drafter needs DROID/sim-scale encoding (open decision 1 below still stands).

## What upstream gives us

- **Encoders**: `torch.hub` entry points in `upstream/src/hub/backbones.py`:
  `vjepa2_vit_large / huge / giant / giant_384` plus V-JEPA 2.1 variants
  (`vjepa2_1_vit_base_384` ... `gigantic_384`).
- **World model**: `vjepa2_ac_vit_giant`, the action-conditioned predictor
  (`src/models/ac_predictor.py`, `forward(x, actions, states, extrinsics=None)`),
  trained on DROID; end-effector actions + proprio states, Franka setting.
- **Planning reference**: `notebooks/energy_landscape_example.ipynb` +
  `franka_example_traj.npz`: goal-latent energy over action candidates; this is
  the "flat CEM" arm of their zero-shot planning results and the anchor we must
  reproduce before changing anything (cube lesson: ANCHOR THE BASELINE FIRST).

## Component mapping (LeWM stack to V-JEPA 2 stack)

| Ours (specaccept/) | LeWM instantiation | V-JEPA 2 instantiation |
|---|---|---|
| latent z | 1-vector per frame (18M DiT space) | token grid per clip; needs a pooling/readout choice before any rel-distance is defined |
| world model rollout | `lewm.rollout` | `VisionTransformerPredictorAC` autoregressive rollout |
| flat planner + cost | CEM over `SubgoalCostModel` | their energy = L1/L2 to goal tokens; CEM as in the energy notebook |
| c* certificate | `cem_flat_cstar` (rel terminal distance of one flat plan) | same read on the AC rollout terminal; needs the pooling choice above |
| drafter (GDM) | DiT diffusion over z-blocks, S=10, k derived | **must be trained**; no upstream analogue; diffusion over pooled latents on DROID/sim trajectories |
| verifier tau=0.20 | criterion-floor derived on PushT, frozen everywhere | frozen at 0.20 first; re-derive from the new env's criterion floor only if the pre-registered transfer fails |

## Open decisions (in order)

1. **Environment/benchmark.** Upstream ships no simulator. Options: (a) offline
   energy-landscape protocol on DROID clips (weakest but cheapest); (b) Franka sim
   (robosuite/ManiSkill) driven through the AC model, closest to their zero-shot
   robot protocol; (c) their exact tabletop tasks if we can stand up hardware-free
   replicas. Decide before any training run.
2. **Latent readout.** Token grid to what the drafter models and c* measures
   (mean-pool vs attentive probe vs token-subset). This choice defines rel-err and
   must be fixed before the criterion-floor / tau-derivability probe.
3. **Anchor.** Reproduce their flat planning number on the chosen setting before
   introducing subgoals (the cube-anchor lesson, non-negotiable).
4. **Drafter training data.** DROID episodes vs sim rollouts; S (stride) in their
   frame-rate units; S=10 was a user decision on LeWM envs, revisit only with data.
5. **Compute.** ViT-g AC rollouts won't fit the ISCA A100 40GB comfortably at CEM
   batch sizes; profile early. Isambard GH200s are the fallback (path dormant,
   cert issue, see memory).

## First concrete steps

- [x] Package implemented + CPU smoke tests passing (2026-07-20): wm wrapper,
      flat-arm pass-through, c* port, both sources, TokenGDM + trainer, policy,
      offline probe. The notebook planning loop is `planner.flat_plan` (upstream
      `cem()` called verbatim, not re-implemented).
- [ ] `pip install -e upstream/` into a fresh venv (needs its own torch pin; do
      NOT share the le-wm venv); run `probe_offline.py` on GPU (ISCA sbatch
      provided); first weights download happens here.
- [ ] Read the probe: anchor sane? c* separating where raw distance saturates?
      floor p50 below tau=0.20? Then pre-register the transfer bars.
- [ ] Environment decision (open decision 1) with the user before any drafter
      training compute is spent.

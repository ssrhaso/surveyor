# FF-JEPA — latent subgoal planner on frozen LeWM

Implements FF-JEPA (arXiv 2606.09311) on top of the **frozen** LeWM world model.
Encoder + predictor are never trained/fine-tuned. Build order: **(A) subgoal
dataset → (B) injection wrapper + oracle test → (C) GDM diffusion planner → (D)
head + N-stride sweep.** GDet is intentionally skipped.

Files:
- `lewm_io.py` — load the frozen LeWM (box: `load_pretrained`; CPU: local
  `config.json`+`weights.pt`), `encode_frames` (192-d latent), the canonical PushT
  target, and `eval_state_tol` (tolerance-parameterized copy of the env criterion,
  verified byte-identical to `PushT.eval_state` at 20°). Reused by all tasks.
- `build_subgoals.py` — **Task A** builder.

---

## Task A — subgoal-dataset builder  ✅ CPU-validated

Per-episode **stride-25 subgoal-latent sequences** from the *successful* expert
PushT episodes, in the frozen encoder's latent space.

### Decisions (configurable; recorded into the output file)
- **Success filter = block-only canonical.** Keep an episode iff its **final
  block pose** reaches the canonical green target `block=(256,256), angle=π/4`
  (env `goal_pose`, fixed because `goal` ∉ `DEFAULT_VARIATIONS`). Uses the env's
  exact `eval_state` math; the agent term of the 4-D `pos_diff` is neutralized
  (target agent = final agent) because PushT has no canonical agent rest pose —
  this matches the paper's "T block ends up in the correct position" prose.
  `eval_state_tol(·,20°)` is asserted byte-identical to `PushT.eval_state`.
- **Headline 20° kept, 5° recorded as a subset flag** (`in_5deg`; 5°-keep ⊆
  20°-keep). Broad sample (n=400): **62.3 % kept @20°, 45.8 % @5°**. NB the
  paper trained the full DM on **8 318** episodes ≈ 44.5 % ≈ our **5 °** number,
  i.e. their filter looks ~5°-grade — our decided headline is 20°, but both masks
  are stored so the GDM stage can pick either.
- **Pure stride-25 subsample** from `ep_offset`: rows `off : off+ep_len : 25`.
  The terminal frame is included only when `(ep_len-1) % 25 == 0` (paper says
  "subsample with stride H"; the eval goal is defined by the eval protocol, not
  this dataset, so we don't force-append it).

### Output schema (`torch.save` dict, ~48 MB full run)
| key | shape / type | meaning |
|---|---|---|
| `latents` | `(Σn_sg, 192) f32` | all kept episodes' subgoals, concatenated in order |
| `lengths` | `(K,) i64` | `n_sg` per kept episode |
| `offsets` | `(K,) i64` | start index into `latents` (`cumsum(lengths)`) |
| `episode_idx` | `(K,) i64` | original dataset episode id |
| `ep_len` | `(K,) i64` | original episode length (frames) |
| `in_5deg` | `(K,) bool` | episode also passes the stricter 5° cut |
| `stride`,`latent_dim` | `25`,`192` | |
| `criterion`,`encoder`,`counts`,`norm_stats`,`sampling` | dicts | provenance |

Reconstruct episode *i*: `latents[offsets[i] : offsets[i]+lengths[i]]`.

### Acceptance (all PASS on CPU, 60-episode random sample)
- `eval_state_tol(·,20°)` == `PushT.eval_state` (60/60).
- Saved per-episode latents == a fresh independent encode (max abs ≈ 1e-6).
- Reload round-trips; counts consistent; `‖z‖` mean **13.94** ≈ √192 = 13.86.

### Run
```bash
# CPU smoke test (Windows, local model) — what was validated:
SDL_VIDEODRIVER=dummy python -m ffjepa.build_subgoals \
  --source local --local-dir ../drift_probe/model \
  --swm-src ../lewm-investigation/stable-worldmodel \
  --h5 ../drift_probe/expert/pusht_expert_train.h5 \
  --out /tmp/subgoals_smoke.pt --max-episodes 60 --sample-mode random --device cpu

# GPU box (A100) — FULL run (all 18,685 episodes, ~63k frames, ~48 MB out):
cd ~/le-wm && STABLEWM_HOME=$HOME/.stable-wm SDL_VIDEODRIVER=dummy \
  python -m ffjepa.build_subgoals \
  --source pretrained --encoder-id quentinll/lewm-pusht \
  --h5 $HOME/.stable-wm/datasets/pusht_expert_train.h5 \
  --out subgoals_pusht.pt --device cuda
```
Expected on the box: `kept@20°≈11,600`, `kept@5°≈8,300`, `‖z‖ mean≈13.9`.

> Code reaches the box by manual paste/upload (the 3 `ffjepa/` files), **not**
> git. Nothing is committed until the end-to-end oracle test (Task B) reproduces
> the baseline at 20° — and only on explicit say-so.

---

## Task B — injection wrapper + oracle test  ✅ CPU-validated

Files: `subgoal_planner.py` (`SubgoalCostModel`, `FFJEPAPolicy`, `OracleSubgoalSource`,
`build_oracle_table`) + `eval_ffjepa.py` (driver).

### Integration mechanism
The subgoal latent travels **inside `info_dict`** as `subgoal_emb`, injected by
`FFJEPAPolicy` after `_prepare_info`. The existing policy-slice (to replanning
envs) + `CEMSolver` expand (across `num_samples`) carry it to `get_cost` aligned
per-env — robust under the solver's `batch_size=1` env batching (a stateful
`set_subgoal` buffer on the cost model would misalign, since `get_cost` never
learns which env it is computing). `FFJEPAPolicy.get_action` mirrors the base
replan detection (buffer empty & not dead), advances each env's subgoal index
(`0→1` on first replan ⇒ target `subgoal[1]`), injects `subgoal_emb`, then
delegates to the base.

`SubgoalCostModel.get_cost`: ignores `info['goal']`, reuses `LeWM.rollout`, returns
**terminal L2²** to the per-env subgoal — `Σ_d (pred_terminal − z)²` over candidates.

### swm version divergence (important)
The **source checkout's** `LeWM.get_cost`/`criterion` are mutually incompatible for
`num_samples>1` (`goal_emb (B,1,D)` cannot `expand_as` `pred (B,S,L,D)`); the box's
pip-installed **0.1.1** (which reproduced the baseline) differs. So `SubgoalCostModel`
computes the terminal L2² **directly** (the spec's wording) rather than calling
`criterion` — version-independent, identical math. It still reuses `LeWM.rollout`.

### Acceptance (all PASS on CPU)
1. **Cost wiring** — `get_cost` == rollout-terminal-L2² to the injected subgoal (Δ=0).
2. **Policy bookkeeping** — replan exactly every 25 steps; `sg_step` `[1,1]`→`[2,2]` at
   step 25; per-env subgoal correctly sliced to replanning envs.
3. **Acceptance-1 (plumbing)** — `oracle subgoal[1]` == baseline goal latent
   (harness `img_transform`→`E`), max |Δ| ≈ 1e-6 across episodes. Built on the
   prior gate that `encode_frames` == the harness goal-encode path (~1e-6).
4. **Full e2e** — `eval_ffjepa.py` runs the *real* harness (World→EnvPool→
   FFJEPAPolicy→SubgoalCostModel→CEM→eval_state pin) error-free on CPU.

### Box results (A100, n=32, seed=42, cem-seed=42)  ✅
| | env-native (4-D agent+block) | block-only (paper criterion) |
|---|---|---|
| **short** 20° / 5° | **90.62% / 81.25%** (== baseline) | — |
| **long**  20° / 5° | 65.62% / 56.25% | **84.38% / 78.12%** |

- **Acceptance-1 (plumbing) ✅** — short oracle reproduces the flat-LeWM baseline
  *exactly* (29/32 @20°, 26/32 @5°). The earlier 84.4% was purely a **CEM-seed**
  mismatch: the driver defaulted `--cem-seed 1234`, but `config/eval/solver/cem.yaml`
  uses `seed: ${seed}` = 42. Proven by `--subgoal baseline` (raw LeWM+goal image
  through this driver) giving the *same* 84.4% at seed 1234 → injection is faithful.
  `--cem-seed` now defaults to `--seed`.
- **Acceptance-2 (ceiling) ✅** — long oracle = the reachability ceiling for GDM.
  Block-only **84.4%** ≈ paper FF-JEPA long **91.8%** (within n=32 noise) vs flat-LeWM
  long collapse **3.5%**. Env-native is stricter (requires the agent near the demo
  goal-frame pose); block-only matches the paper ("block within 20px + angle").
- **Scoring = report both** (`--score both`, default): env-native (baseline-comparable)
  + block-only (paper-comparable). GDM will be measured against the oracle under the
  same `--score`.
- Cost contract verified on 0.1.1: `SubgoalCostModel.get_cost` == `LeWM.get_cost`
  (Δ=0) given the same goal latent; sampling byte-identical to `eval.py`; encode path
  matches harness to ~1e-6. (Source-checkout `get_cost` only mis-broadcasts at B≥2,
  never hit — CEM runs `batch_size=1`.)

### Run
```bash
# CPU smoke (validated): tiny CEM, 2 envs, local model
SDL_VIDEODRIVER=dummy python -m ffjepa.eval_ffjepa --source local \
  --local-dir ../drift_probe/model --swm-src ../lewm-investigation/stable-worldmodel \
  --h5 ../drift_probe/expert/pusht_expert_train.h5 --device cpu \
  --num-eval 2 --num-samples 16 --n-steps 3 --topk 4 --mode short --eval-budget 25 --angles 20

# Box (A100): oracle short, n=32, baseline-comparable episode set (seed 42), both tolerances
cd ~/le-wm && STABLEWM_HOME=$HOME/.stable-wm SDL_VIDEODRIVER=dummy \
  python -m ffjepa.eval_ffjepa --source pretrained --encoder-id quentinll/lewm-pusht \
  --h5 $HOME/.stable-wm/datasets/pusht_expert_train.h5 --device cuda \
  --num-eval 32 --mode short --angles 20 5
# then --mode long for the ceiling.
```

### Next: Task C
Train GDM (DiT diffusion planner, WG=1, N=3) on the Task-A `subgoals_pusht.pt`,
add a `GDMSubgoalSource` (plug into `--subgoal gdm`), evaluate GDM SR / oracle SR.

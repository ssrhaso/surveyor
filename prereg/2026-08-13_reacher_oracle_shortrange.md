# P-ORC: is the short-range Reacher tax the waypoint structure, or the drafts?

**Frozen 2026-08-13, before any cell below was run.** Motivated by review
preparation: the paper's regime map attributes the short-range collapse of
every waypoint arm on Reacher (`t=25`: adaptive 55.66, `d=1` every-step 65.04,
against flat ~86-96 depending on population and window) to the *decomposition
tax inside the planner's lookahead*, but the ground-truth-waypoint bound behind
that sentence was measured on PushT only (oracle waypoints tie flat, 84.2 vs
84.0). On Reacher the tax explanation is currently inference, not measurement.

## The question

At short range, do **ground-truth** waypoints (the episode's own future
latents, `S=10`) collapse the way drafted waypoints do?

* **Reading A (draft noise).** Oracle waypoints recover flat. The tax is the
  noise of *sampled* drafts pursued as targets, not waypoint structure; PushT
  and Reacher then agree.
* **Reading B (structural).** Oracle waypoints collapse toward the drafted
  arms. Decomposition itself detours at short range, even with perfect
  waypoints.

**Frozen prediction: Reading A.** The PushT tax cell measured ground-truth
waypoints *tying* flat inside the window, and nothing in the regime map
predicts an environment flip of that bound. Stated so the convenient
after-the-fact story cannot be chosen to fit.

## Design

One array, six tasks, so every comparison is paired in-batch on identical
episodes and seeds. Population `reacher_c2222.ep100.s2222.json` over
`/lustre/home/ha676/data/reacher/reacher.h5`, `n=128`, seeds 42-45, budget
`2t`, joint criterion, byte-identical to the banked depth/adaptive arms (jobs
2331806/2334396) except the subgoal source and, for the flat arms, the window:

* **oracle `S=10`** (`--subgoal oracle --stride 10`), `t in {25, 50}`;
* **flat RH2** (`--subgoal baseline --horizon 2 --receding-horizon 2`),
  `t in {25, 50}`;
* **flat RH5** (`--subgoal baseline --horizon 5 --receding-horizon 5`),
  `t in {25, 50}`.

Banked comparators quoted, not re-run: adaptive `tau=0.20` 55.66/79.49 and
`d=1` every-step 65.04/87.69 at `t=25/50` (same population, seeds and budget).

**Caveat recorded before the run.** `OracleSubgoalSource` is a static lookup
(subgoals precomputed from the episode's own trajectory at absolute stride
offsets; it never re-anchors to the realized state). At `t=25/50` that is 2-3
hops, so staleness is small but biases the oracle *down*. Reading B can
therefore be partially staleness; Reading A cannot be produced by it.

## Frozen predictions

**P-ORC-0 (the phenomenon reproduces in-batch).** The strongest in-batch flat
arm leads the banked adaptive arm by >= 10pp at `t=25`. *If this fails, the
batch does not exhibit the collapse and P-ORC-1 is not scored.*

**P-ORC-1 (the split).** Oracle waypoints land within **2.0pp** of the
strongest in-batch flat arm at `t=25` (Reading A). *Refuted if oracle trails
by more than 2.0pp.*

**P-ORC-2 (the depth of a refutation).** If P-ORC-1 is refuted and oracle
falls within 2.0pp of the `d=1` every-step arm (65.04), Reading B is full:
decomposition itself collapses. Between the two bands, the result is reported
as **bounded** (ground-truth waypoints cost between 2pp and ~20pp, part
attributable to staleness) and no mechanism sentence is changed.

## Consequence rules (frozen before the run)

* **P-ORC-1 holds (Reading A).** The paper's mechanism account of short-range
  Reacher gains one measured clause: the tax is the noise of sampled drafts
  (ground-truth waypoints recover flat), matching PushT; the routing remedy is
  unchanged. The two-part decomposition of the collapse (structural tax +
  transferred-tau under-re-anchoring) is restated as draft-noise +
  under-re-anchoring.
* **P-ORC-1 refuted, P-ORC-2 full (Reading B).** The regime-map sentence
  gains the Reacher measurement as *structural*: even perfect waypoints
  detour inside the window, with the staleness caveat stated.
* **Bounded outcome.** Reported in the prereg and appendix only; no main-text
  mechanism sentence changes.
* In every case the deployed policy is unaffected: the router already sends
  these cells to flat/GC-IDM.

No cell is excluded after the fact. A run producing no SR line is re-run once
with the identical command and otherwise reported missing.

## Runner

`batch/isca/run_reacher_oracle_short.sbatch` (array 0-5).

## Amendment: a builder artifact voids the first `t=25` oracle cell (2026-08-13, recorded before any fix ran)

Job 2335860 completed all six cells. Harvest (means, seeds 42-45):

| arm | `t=25` | `t=50` |
|---|---|---|
| oracle `S=10` | 33.40 | 99.22 |
| flat RH2 | 95.70 | 86.53 |
| flat RH5 | 82.81 | 92.97 |

The `t=25` oracle number is an artifact, and the evidence is the builder, not
the SR. `build_oracle_table` sizes the table as
`n_sg = goal_offset // stride + 1` frames at `start + k*stride`, so at
`t=25`, `stride=10` it encodes `[start, +10, +20]` and **never encodes the
goal frame at +25**; the arm's log prints `K = 3` against `t=50`'s `K = 6`
(which ends on `+50 = goal` because 50 divides evenly). `OracleSubgoalSource`
clamps its index to the last row, so the `t=25` arm parks on the exploratory
state at `+20` for the rest of the episode and is never given the goal as a
target. Its 33.40 measures how often `qpos(+20)` happens to sit within 0.05
rad of `qpos(+25)`, not decomposition.

**Stated against ourselves, before the corrected run:** the voided cell was
*adverse* to the frozen Reading-A prediction, so voiding it favours our own
prediction. That is why the invalidation rests on the builder line and the
`K` print rather than on the result, and why the corrected cell may still
refute Reading A, in which case it does and is scored as such.

**The fix, and why it touches nothing banked.** `build_oracle_table` gains a
conditional append of the goal frame when `goal_offset % stride != 0`. Every
banked oracle cell used divisible pairs (PushT stride 25 at `t=25/75/150`),
where the append is a no-op by construction; the `t=50` cell here is likewise
divisible, byte-unaffected, and is not re-run.

**Gate for the corrected run, fixed now:** the re-run `t=25` oracle log must
print `K = 4`; otherwise the cell is void again. Bars and consequence rules
above are unchanged.

## Outcome (corrected run, job 2335868; K=4 gate PASSED)

Corrected `t=25` oracle cell, seeds 42-45: 98.44/97.66/99.22/97.66, mean
**98.24**. Full picture on identical episodes (`t=25` / `t=50`):

| arm | `t=25` | `t=50` |
|---|---|---|
| oracle `S=10` (goal-terminated) | **98.24** | **99.22** |
| flat RH2 | 95.70 | 86.53 |
| flat RH5 | 82.81 | 92.97 |
| every-step drafted, `d=1` (banked) | 65.04 | 87.69 |
| accept rule `tau=0.20` (banked) | 55.66 | 79.49 |

### Verdicts against the frozen predictions

* **P-ORC-0: PASSES** (strongest in-batch flat leads the banked adaptive arm
  by 40.0pp at `t=25`).
* **P-ORC-1: HOLDS, Reading A, stronger than predicted.** Oracle waypoints do
  not merely land within 2.0pp of the strongest flat arm; they lead it, by
  +2.54pp at `t=25` and +6.25pp at `t=50`. Ground-truth waypoints cost
  nothing at short range.
* **P-ORC-2: moot.**

One mechanism note, reported not claimed: the oracle's edge over flat is
expected from hindsight-goal structure (the goal is the endpoint of the very
trajectory whose waypoints the oracle serves, so tracking them is a
guaranteed route given budget); the arm uses privileged future frames and is
a bound, not a deployable policy.

### Consequence applied (Reading A)

The short-range Reacher collapse is **draft noise, not waypoint structure**:
perfect waypoints at the same `S=10` beat flat while sampled drafts collapse
by 30-40pp, with the transferred `tau` adding ~9pp of under-re-anchoring on
top (P-TAU / Extension A). This matches the PushT tax cell (ground-truth
waypoints tie flat, 84.2 vs 84.0), so the regime-map account is now measured
on both dev environments. Applied to `main_workshop_final.tex`: the regime
map's tax parenthesis gains the Reacher numbers, `sec:results-executor` names
the tax as draft noise, and `app:negatives` gains the full table with the
voided-cell disclosure. The routing remedy is unchanged.

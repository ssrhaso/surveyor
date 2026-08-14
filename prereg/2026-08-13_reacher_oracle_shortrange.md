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

## Outcome

*(appended after the runs)*

# Pre-registration: baseline battery (random floor, GC-IDM, cited comparators)

**Written:** 2026-08-07T11:13Z (12:13 local, Europe/London)
**Scope:** the baseline block requested 2026-08-07 - random-action floor
(job 2329041), GC-IDM as a flat comparator, DINO-WM horizon panel, and the
transcription of LeWM Fig. 6.
**Rule:** every prediction below is frozen at the timestamp on its own heading.
Outcomes are reported as-is whether they confirm or refute. No bar is amended
after a readout.

---

## A. HONESTY NOTE ON THE RANDOM-FLOOR PREDICTION - READ FIRST

The Cube random-floor prediction (P-RF-1) was requested "before job 2329041
reports", void if written after the results land. **It was written after the
results landed, and is therefore NOT a pre-registration.** The facts, recorded
exactly:

* Array 2329041 was submitted 2026-08-07 ~11:17 local and ran far faster than
  estimated (random plans nothing: no CEM solve, no drafter). Per `sacct`, task 0
  completed 11:54:20; the Cube cells (tasks 52-59) completed between
  **12:11:02 and 12:11:46 local**.
* A queue-status check at **12:11:22 local** returned a count of log files
  containing the string `SR`. That check counted filenames only
  (`grep -l ... | wc -l`); it returned no success rates.
* This file was created at 12:13 local, i.e. **after** the Cube cells finished.

What is therefore still true and worth recording: **the author had not read any
success-rate value from this array at the time of writing.** P-RF-1 below is
consequently downgraded to a **blind interpretation rule** - a decision rule
fixed while blind to the number, which is weaker than pre-registration (the
result existed and could in principle have been consulted) but stronger than a
post-hoc reading. It is labelled as such wherever it appears and must never be
described as pre-registered.

P-GCIDM-1, P-GCIDM-2 and P-DISP-1 are unaffected: neither GC-IDM nor the
displacement probe had been implemented or run when this file was written, so
those are genuine pre-registrations.

---

## P-RF-1 (BLIND INTERPRETATION RULE - NOT PRE-REGISTERED) - Cube random floor

**Frozen blind to the value at:** 2026-08-07T11:13Z

LeWM Fig. 6 reports Random at **48%** on OGBench-Cube under their protocol. Our
Cube cell is a different protocol: goal offset 150, budget 300, success = cube
within 0.04 m, `--episode-min 8000`, n=128/seed over seeds 42-49.

**Expectation:** our Cube random floor lands in **single digits to low teens**,
operationalised as **SR in [0, 15]%** pooled over the eight seeds.

**Declared falsifier, fixed in advance of reading:** if the pooled Cube random
floor returns **>= 30%**, then our Cube protocol is *not* as separated from
LeWM's as we have been claiming, and the held-out-environment framing for Cube
needs revisiting - specifically the claim that Cube is a genuinely harder,
independently-constructed protocol rather than a relabelling of theirs.

This is a **bar over-specification**, not a goalpost move: the [0,15] band and
the >=30% trigger are stated together, and the interval between them (15-30%) is
declared in advance to be an ambiguous outcome that supports neither reading and
will be reported as such.

---

## P-DISP-1 (PRE-REGISTERED) - Cube displacement probe

**Frozen at:** 2026-08-07T11:13Z. Probe not yet written or run at this time.

Offline pass over cached Cube episodes, no policy involved. Measure cube
displacement between start frame and goal frame under (a) LeWM's Fig. 6 Cube
protocol and (b) our certified protocol at goal offset 150. Report median and
p90 for each, and the **fraction of LeWM-protocol episodes whose start-to-goal
displacement is below the 0.04 m success radius**.

**Prediction:** that fraction is **large** - operationalised as **>= 30%** of
LeWM-protocol episodes starting already inside the success radius - which would
be the mechanism explaining how a random policy reaches 48% there.

**Consequence, declared now:** this number gates the citation block. If the
fraction is large, LeWM Fig. 6's Cube column may be cited only with the
displacement caveat attached, because its Random entry is then measuring
protocol geometry rather than task difficulty. If the fraction is small
(< 10%), the hypothesis is refuted, the 48% needs a different explanation, and
we say so.

---

## P-GCIDM-1 (PRE-REGISTERED) - GC-IDM may displace LeWM-flat as "strongest flat"

**Frozen at:** 2026-08-07T11:13Z. GC-IDM not implemented in this repo at this
time; no GC-IDM cell has been run.

The headline grid's margin row is defined against *the strongest flat arm*, not
against LeWM-flat specifically. GC-IDM (arXiv 2605.08732) is a flat controller
on our own substrate (frozen LeWM encoder, ~1.5M-param MLP, AdaLN-Zero horizon
conditioning).

**Declared rule:** if GC-IDM exceeds LeWM-flat in **any** cell, then in that
cell GC-IDM **becomes the strongest-flat reference** and the margin row is
recomputed against it. This applies cell by cell, automatically, with no
discretion at readout, and applies even where it shrinks our reported margin.
Cells affected are listed explicitly in the readout.

Grid cells in scope: PushT t{25,50,75,100,150}, Reacher t{25,50,100,150}
(max-offset-100 and max-offset-150 populations kept unspliced), Cube t=150.

---

## P-GCIDM-2 (PRE-REGISTERED) - GC-IDM collapses at long horizon

**Frozen at:** 2026-08-07T11:13Z.

**Prediction:** GC-IDM collapses at long horizon in the same manner as flat CEM,
for the same structural reason - a single forward pass with no lookahead cannot
represent a goal that outruns the planning window.

**Operationalisation:** GC-IDM's SR falls monotonically with goal offset on
PushT and is **below its own t=25 cell by at least 40pp at t=150**.

**Declared falsifier:** if GC-IDM does **not** collapse - if it holds within
20pp of its short-horizon cell at t=150 - that is **headline-affecting**, because
it would mean a cheap flat controller achieves what we attribute to verified
subgoal consumption. It is reported as-is, prominently, not buried. In that case
P-GCIDM-1 also applies and the margin row is recomputed against it.

**Cost, reported alongside SR (no prediction frozen):** GC-IDM claims 100-130x
cheaper planning. Call cost per decision is recorded for every cell so the
comparison lands on the same footing as the existing cost accounting.

---

---

# READOUTS

## P-DISP-1 READOUT - CONFIRMED (2026-08-07T11:34Z)

`surveyor/probes/probe_cube_displacement.py`, 10,000 episodes (all ep_len 201),
`Results/cube_displacement.json`. Displacement = `||block_pos[start+offset] -
block_pos[start]||` in metres; radius 0.04 m.

| protocol | n pairs | median (m) | p90 (m) | < 0.04 m |
|---|---:|---:|---:|---:|
| LeWM Fig. 6 (offset 25, uniform start, all valid rows) | 1,760,000 | 0.0959 | 0.2836 | **38.38%** |
| LeWM protocol, seed-42 draw of n=50 (their num_eval) | 50 | 0.1269 | 0.2691 | 34.00% |
| ours (offset 150, start = ep_len-1-150, episodes >= 8000) | 2,000 | 0.2671 | 0.3194 | **0.00%** |
| ours, all episodes (no --episode-min) | 10,000 | 0.2685 | 0.3195 | 0.00% |

**38.38% >= the declared 30% trigger: CONFIRMED.** Under LeWM's protocol more
than a third of episodes begin with the cube already inside the success radius,
so a policy that does nothing succeeds on them; that is the mechanism behind
their Random entry of 48%. Under our protocol the figure is **0.00%** - not one
episode of 10,000 is vacuous, with or without the held-out episode floor, so the
floor is not what creates the separation.

**Consequence, as declared in advance:** LeWM Fig. 6's Cube column may be cited
only with this caveat attached. The two Cube protocols are not comparable, and
their Random entry measures protocol geometry rather than task difficulty.

## P-RF-1 READOUT - expectation met, at the top edge of the band (2026-08-07T11:36Z)

Random floor, job 2329041, pooled over the declared seeds, all cells at full
seed count (`Results/random_floor.csv`).

**Cube t=150 = 14.84%** (152/1024, seeds 42-49).

Against the blind rule: the declared band was [0, 15]% and the falsifier >= 30%.
The falsifier is **not** triggered, so the Cube protocol separation claim stands
and the held-out-environment framing does not need revisiting. But 14.84 sits at
the **very top edge** of the declared band, and that is recorded rather than
smoothed: the expectation was met, not comfortably.

Read together with P-DISP-1 the two numbers are coherent and mutually
reinforcing: 0.00% of our Cube episodes start inside the success radius, so the
whole 14.84% is random actions genuinely nudging the cube into tolerance within
300 steps, not vacuous cells. LeWM's 48% decomposes very differently - 38.38%
vacuous starts plus a smaller genuine remainder.

## Random floor, full grid (context for the whole block)

| env | t | seeds | succ/n | SR% |
|---|---:|---:|---:|---:|
| PushT (20 deg) | 25 | 4 | 847/1024 | 82.71 |
| PushT (20 deg) | 50 | 4 | 392/1024 | 38.28 |
| PushT (20 deg) | 75 | 4 | 104/1024 | 10.16 |
| PushT (20 deg) | 100 | 4 | 33/1024 | 3.22 |
| PushT (20 deg) | 150 | 4 | 1/1024 | 0.10 |
| Reacher | 25 | 8 | 130/1024 | 12.70 |
| Reacher | 50 | 8 | 129/1024 | 12.60 |
| Reacher | 100 | 8 | 94/1024 | 9.18 |
| Reacher | 150 | 8 | 105/1024 | 10.25 |
| Cube | 150 | 8 | 152/1024 | 14.84 |
| Two-Room | 25 | 12 | 173/768 | 22.53 |
| Two-Room | 75 | 12 | 29/768 | 3.78 |

PushT 5 deg is carried in the CSV (70.80 / 22.27 / 3.52 / 0.78 / 0.10).

**Two observations recorded now, before any figure or table is built, because
both cut against us and must not be discovered later:**

1. **PushT t=25 has a random floor of 82.71%.** The short-horizon PushT cells
   are therefore weakly discriminative: our +1.5pp margin over LeWM-flat at
   t=25 sits on top of a task a random policy already solves four times in five.
   This is an argument *for* separating short from long horizon in the results,
   not against it - random falls 82.71 -> 0.10 across t=25 -> 150 while our arm
   holds ~98-99 - but the t=25 cell should never be presented as evidence of
   anything on its own.
2. **Reacher's random floor is horizon-independent at ~9-13%**, and it
   independently reproduces the "random-policy floor (8.6-11.3)" already quoted
   in the paper for the goal-free Reacher drafter. That claim now has a directly
   measured basis rather than an inferred one.

---

## Reporting discipline for this block

* Pool numerator and denominator across seeds; never average percentages.
* Any cell short of its declared seed count is flagged, not silently pooled.
* DINO-WM cells are **directional only** (n_evals=10, ~5-6pp per-cell SE): they
  may support "flat collapses on a second architecture too" and may never carry
  a paired margin. Any cell hitting walltime is flagged as truncated, the same
  failure mode that voided the banked goal_H=15 pair.
* No PLDM port. LeWM Fig. 6 is transcribed and cited, each row carrying its own
  protocol and source, following Sub-JEPA (2605.09241) and FF-JEPA, both of
  which cite rather than reimplement because official checkpoints are
  unavailable.

# P-HMAX-1: is there any H_max that rescues GC-IDM at long horizon?

**Written:** 2026-08-07T14:32Z (15:32 local)
**Status: GENUINE PRE-REGISTRATION.** No cell of this sweep has been run. The
array is submitted only after this file is saved. (Written before submission
this time, per the rule adopted in `2026-08-07_gcidm_branches.md` after the
window closed twice today while jobs were already in flight.)

---

## Motivation, from two measurements already banked

GC-IDM conditions on a normalised remaining horizon capped at `H_max`, and the
paper fixes `H_max = T` (the evaluation budget). At our long-horizon cell
(t=150, budget T=300) we now have two points:

| config | PushT t=150 |
|---|---:|
| `H_max=50`, served at T=300 (clamped; breaks their rule) | 63.38 |
| `H_max=300 = T` (their rule, correct port) | **57.81** |

Both are far below our arm's 98.1. The two failures have *different* causes:
clamping feeds an out-of-distribution horizon signal, while widening `H_max`
forces one 1.5M-parameter model to regress actions for goal pairs up to 300
steps apart. The paper's own Table 3 shows the second effect in miniature
(PushT: `H_max=50` → 85.0, `H_max=100` → 83.5, at budget 50).

**Hypothesis (H-TRADEOFF):** on a task whose difficulty genuinely scales with
horizon, amortised inverse dynamics faces a bind with no good setting — small
`H_max` incurs clamping shift, large `H_max` dilutes capacity across a harder
regression problem. Subgoal decomposition escapes it by construction, because
each waypoint is a short-horizon problem no matter how long the episode is.

## P-HMAX-1 (PRE-REGISTERED)

Serve the five already-trained PushT checkpoints
`gcidm_pusht_h{50,100,150,200,300}.pt` at the **same** cell: t=150, budget 300,
the banked `pusht.episodes150s5.json` population (n=256), 20 deg criterion.
One seed: the arm is deterministic on this population (proved by v1's
byte-identical replicas), so seeds are replicas and n_eff = 256.

**Prediction:** no setting of `H_max` rescues it. Operationalised as
**max over the five cells < 75.0%**, against our arm's 98.1.

**Declared falsifier:** if any `H_max` reaches **>= 90.0%**, the long-horizon
collapse is a configuration artifact rather than a property of the approach, our
PushT margin is not real, and that is reported as the headline correction.

**Declared ambiguous band:** a maximum in [75, 90) supports neither reading. It
would mean `H_max` matters more than we claim but does not close the gap, and it
is reported as such rather than spun either way.

**Secondary, no prediction frozen:** the *shape* of SR vs `H_max` is recorded
(monotone, U-shaped, or flat). Shape is descriptive here; only the maximum
carries the verdict.

## Reporting rules

* Quote `n_eff = 256`, never a pooled multiple of it.
* The comparison against our arm uses the **best** cell in this sweep, so the
  baseline is given its most favourable configuration.
* Both prior points (63.38 at clamped `H_max=50`, 57.81 at `H_max=300`) stay on
  the record alongside the sweep; nothing is quietly replaced.
* Outcome reported as-is whichever way it falls.

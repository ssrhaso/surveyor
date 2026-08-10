# GC-IDM branch registration - Reacher and Cube

**Written:** 2026-08-07T12:24Z (13:24 local, Europe/London)
**Companion to:** `prereg/2026-08-07_baselines.md` (P-GCIDM-1, P-GCIDM-2)

---

## A. STATUS - THIS IS NOT A PRE-REGISTRATION. READ FIRST.

This file was requested "before Reacher t=100/150 land", void once the cells
report. **The cells had already reported.** Facts, exactly:

* GC-IDM grid array 2329277 (60 cells) completed between
  **2026-08-07T12:49:16 and 13:13:55 local**. All 60 COMPLETED.
* This file was written at **13:24 local**, roughly ten minutes after the last
  cell finished and ~35 minutes after the first.
* At the time of writing the author had inspected only job states and a count of
  log filenames. **No success-rate value from any Reacher t=100/t=150 or Cube
  grid cell had been read.**

So, exactly as with P-RF-1 in the companion file, everything below is a **blind
interpretation rule** - a decision rule fixed while blind to the numbers - and
**not** a pre-registration. It is weaker (the data existed and could in
principle have been consulted) and stronger than a post-hoc reading. It must
never be described as pre-registered.

**This is the second time in one day the window has closed before the file was
written.** The cause is the same both times: these arrays finish far faster than
estimated (GC-IDM plans nothing - one MLP forward per step), so "write the
prereg while the jobs run" has failed twice. The working rule from here is that
a prediction file is written and saved *before the array is submitted*, not
while it is in flight. Recorded so the failure is not repeated a third time.

---

## B. THE MECHANISM HYPOTHESIS - AND WHY ITS VALUE IS NOW LOST

The hypothesis, stated for the record:

> GC-IDM collapses where the task requires **sequencing** beyond the planning
> window, and holds where distance grows without compositional structure -
> **transport** to a static target that the goal latent already encodes.
> PushT is contact-rich routing (confirmed collapse, 100.0 -> 63.4). Reacher is
> transport to a static target.

This is a sharper axis than the paper's current "beyond the planning window",
and it would be a genuine contribution. **But it is only usable as a prediction
if it predates the data, and it does not.** Written after the Reacher cells
existed, it can be offered as a post-hoc *explanation* and nothing more. Any
sentence of the form "we predicted this" is unavailable to us for Reacher and
Cube.

**Salvage, and the only way this axis becomes claimable:** it must be tested
prospectively on a cell that has not been run. The clean candidate is
**Two-Room** - navigation that requires routing around a wall, i.e. sequencing,
on an environment where GC-IDM has never been trained or evaluated (no stride-1
dense latent cache exists for it yet). The frozen prediction, registered here
and now, genuinely in advance:

> **P-MECH-1 (PRE-REGISTERED, no Two-Room GC-IDM cell exists at write time):**
> On Two-Room, GC-IDM collapses at long horizon in the PushT manner rather than
> holding in the Reacher manner, because crossing between rooms requires routing
> around an obstacle and cannot be executed as monotone transport toward a goal
> latent. Operationalised: GC-IDM at t=75 falls at least 25pp below its own
> t=25 cell. Falsifier: it holds within 10pp of its t=25 cell, in which case
> the sequencing/transport axis is refuted and must be dropped, not rescued.

This requires a Two-Room dense latent build plus one GC-IDM training run. It is
**proposed, not executed** - the standing instruction is that nothing further
needs running, so this is the user's call. Absent that run, the
sequencing/transport axis is reported as a post-hoc observation and explicitly
labelled as one.

---

## C. REACHER BRANCHES (blind interpretation rules)

Our arm (\method{}): **92.3 at t=100, 90.1 at t=150.**
GC-IDM's own short-horizon cell on our populations: **100.0 at t=25**, 99.22 at
t=50 (both already read before this file, and both already losses for us:
-1.3pp and -2.9pp).

**Branch A - GC-IDM lands below our arm at both t=100 and t=150.**
PushT's collapse replicates on a second environment. The regime map is confirmed
on two environments rather than one, and the paper's central claim strengthens:
amortised control wins inside the planning window and fails beyond it, measured
against a published, 100-130x cheaper controller.

**Branch B - GC-IDM holds within a few points of its own t=25 cell (100.0).**
Then **Reacher is a loss and is reported as one**, plainly, in the table and in
the text. The headline narrows to PushT and Cube. Under this branch the
sequencing/transport story fits well - but per section B we do not get to claim
we predicted it, and the fit of an explanation is not evidence for it.

**Stated in advance so it cannot be walked back:** Branch B is not a goalpost
move. If GC-IDM beats us on Reacher at long horizon, that is a loss on that
environment and is reported as a loss **regardless of how neatly the mechanism
story accommodates it**. An explanation that arrives after the result does not
convert a loss into a finding.

Boundary case, declared: if GC-IDM lands within +/-1pp of our arm at a horizon,
that cell is reported as a tie, not claimed either way.

---

## D. CUBE BRANCHES (blind interpretation rules)

Our arm: **79.9 at t=150** (goal offset 150, budget 300, criterion = cube within
0.04 m). Same two branches as Reacher, with the same rule that Branch B is a
reported loss.

**Two constraints on how the Cube cell may be read, both measured in advance:**

1. **GC-IDM's published Cube 99.3 is NOT comparable to our t=150 number.** It is
   the offset-25 protocol, on which P-DISP-1 measured that **38.38%** of
   episodes begin with the cube already inside the success radius. That column
   is against a high floor and a different task.
2. **Read the t=150 cell against our measured random floor of 14.84**, not
   against zero. Cube's criterion constrains only cube position, so the
   meaningful question is the distance above 14.84, not the raw percentage.

A GC-IDM Cube number that looks strong in isolation must be checked against both
before any claim is made about it.

---

## E. WHAT IS UNAFFECTED

P-GCIDM-1 (GC-IDM displaces LeWM-flat as the strongest non-drafting baseline
wherever it beats it, and the margin row is recomputed against it, even where
that shrinks our margin) was frozen this morning, before any GC-IDM cell ran,
and is a genuine pre-registration. It applies to every cell in this file
automatically and without discretion.

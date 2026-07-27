"""Deployed as dino_wm/validate_tworoom_env.py.

Anchors the vendored TwoRoom eval env against the recorded dataset BEFORE any
closed-loop cell runs (the cube lesson: never trust a number from an
un-anchored substrate). Four checks:

  R  renderer   -- render at each recorded agent position, compare pixel-wise
                   against the stored frame from tworoom.h5
  D  dynamics   -- (pos_t, action_t) -> pos_{t+1} must reproduce the recording,
                   including wall/door collisions and border clamping
  A  actions    -- recorded action range (confirms the [-1,1] x speed contract)
  P  population -- opposite-room fraction and door-routed path length in the
                   data, which is what sample_random_init_goal_states must match

Exit code 0 only if R and D pass at tolerance; the battery scripts depend on it.
"""
import sys

sys.path.insert(0, "/lustre/home/ha676/dino_wm")
import os  # noqa: E402

os.chdir("/lustre/home/ha676/dino_wm")

import hdf5plugin  # noqa: F401,E402  (registers HDF5 compression filters)
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.tworoom.tworoom_env_wrapper import (  # noqa: E402
    AGENT_SPEED, DOOR_POSITION, SUCCESS_RADIUS, WALL_CENTER, TwoRoomEnvWrapper,
)

H5 = "/lustre/home/ha676/data/tworoom/tworoom.h5"
N_EPS = 12          # episodes sampled for the pixel/dynamics checks
N_FRAMES = 8        # frames per episode for the pixel check
PIX_TOL = 1.0       # mean abs pixel difference we will accept
POS_TOL = 1e-3      # px

env = TwoRoomEnvWrapper()
fails = []

with h5py.File(H5, "r") as f:
    off = np.asarray(f["ep_offset"][:])
    ln = np.asarray(f["ep_len"][:])
    agent = np.asarray(f["pos_agent"][:], dtype=np.float32)
    target = np.asarray(f["pos_target"][:], dtype=np.float32)
    act = np.asarray(f["action"][:], dtype=np.float32)
    px = f["pixels"]

    rng = np.random.RandomState(0)
    eps = rng.choice(len(off), size=N_EPS, replace=False)

    # ---- A: action contract ------------------------------------------------
    valid = ~np.isnan(act).any(axis=-1)
    a_ok = act[valid]
    print("[A] action min/max: %.4f / %.4f  (expect within [-1, 1])"
          % (a_ok.min(), a_ok.max()))
    print("[A] |action| p50/p99: %.4f / %.4f"
          % (np.percentile(np.abs(a_ok), 50), np.percentile(np.abs(a_ok), 99)))

    # ---- R: renderer -------------------------------------------------------
    print("\n[R] renderer vs recorded pixels")
    worst = 0.0
    for e in eps:
        s, L = int(off[e]), int(ln[e])
        idxs = np.linspace(0, L - 1, N_FRAMES).astype(int)
        for t in idxs:
            recorded = np.asarray(px[s + int(t)])            # (H, W, C) uint8
            ours = env._render_agent(agent[s + int(t)])
            mad = float(np.abs(recorded.astype(np.int16)
                               - ours.astype(np.int16)).mean())
            worst = max(worst, mad)
    print("    worst mean-abs-pixel-diff over %d frames: %.4f (tol %.2f)"
          % (N_EPS * N_FRAMES, worst, PIX_TOL))
    if worst > PIX_TOL:
        fails.append("renderer mean-abs-diff %.4f > %.2f" % (worst, PIX_TOL))

    # ---- D: dynamics -------------------------------------------------------
    print("\n[D] dynamics vs recorded transitions")
    worst_pos = 0.0
    n_checked = 0
    n_collide = 0
    for e in eps:
        s, L = int(off[e]), int(ln[e])
        for t in range(L - 1):
            a = act[s + t]
            if np.isnan(a).any():
                continue
            env.agent_position = torch.as_tensor(agent[s + t],
                                                 dtype=torch.float32)
            _, _, _, info = env.step(a)
            pred = info["proprio"]
            true = agent[s + t + 1]
            d = float(np.linalg.norm(pred - true))
            # did the recording itself hit a constraint on this step?
            free = agent[s + t] + np.clip(a, -1, 1) * AGENT_SPEED
            if np.linalg.norm(free - true) > 1e-3:
                n_collide += 1
            worst_pos = max(worst_pos, d)
            n_checked += 1
    print("    checked %d transitions (%d of them constrained by wall/border)"
          % (n_checked, n_collide))
    print("    worst position error: %.6f px (tol %.4f)" % (worst_pos, POS_TOL))
    if worst_pos > POS_TOL:
        fails.append("dynamics worst error %.6f > %.4f" % (worst_pos, POS_TOL))

    # ---- P: population -----------------------------------------------------
    print("\n[P] dataset init/goal population")
    n_look = 2000
    a0 = agent[off[:n_look].astype(int)]
    t0 = target[off[:n_look].astype(int)]
    opposite = ((a0[:, 0] < WALL_CENTER) != (t0[:, 0] < WALL_CENTER))
    print("    opposite-room fraction: %.4f" % opposite.mean())
    door = np.array([WALL_CENTER, DOOR_POSITION], dtype=np.float32)
    path = (np.linalg.norm(a0 - door, axis=1)
            + np.linalg.norm(t0 - door, axis=1))
    print("    door-routed path length p10/p50/p90: %.1f / %.1f / %.1f px"
          % tuple(np.percentile(path, [10, 50, 90])))
    print("    => steps at speed %.1f: p10/p50/p90 = %.1f / %.1f / %.1f"
          % ((AGENT_SPEED,) + tuple(np.percentile(path, [10, 50, 90])
                                    / AGENT_SPEED)))
    straight = np.linalg.norm(a0 - t0, axis=1)
    print("    straight-line dist p50: %.1f px (success radius %.1f)"
          % (np.median(straight), SUCCESS_RADIUS))

    # ---- our sampler, same summary ----------------------------------------
    print("\n[P] sample_random_init_goal_states population (seeds 1..2000)")
    inits, goals = [], []
    for sd in range(1, n_look + 1):
        i, g = env.sample_random_init_goal_states(sd)
        inits.append(i)
        goals.append(g)
    inits = np.stack(inits)
    goals = np.stack(goals)
    opp = ((inits[:, 0] < WALL_CENTER) != (goals[:, 0] < WALL_CENTER))
    print("    opposite-room fraction: %.4f" % opp.mean())
    path2 = (np.linalg.norm(inits - door, axis=1)
             + np.linalg.norm(goals - door, axis=1))
    print("    door-routed path length p10/p50/p90: %.1f / %.1f / %.1f px"
          % tuple(np.percentile(path2, [10, 50, 90])))
    print("    => steps at speed %.1f: p10/p50/p90 = %.1f / %.1f / %.1f"
          % ((AGENT_SPEED,) + tuple(np.percentile(path2, [10, 50, 90])
                                    / AGENT_SPEED)))

print("\n" + "=" * 60)
if fails:
    print("VALIDATION FAILED:")
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("VALIDATION PASSED - vendored TwoRoom env matches the recorded data")

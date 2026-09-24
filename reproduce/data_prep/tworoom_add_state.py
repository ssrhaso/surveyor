"""Add the `state` / `goal_state` columns the code reads to the Hugging Face Two-Room h5.

The file ships `pos_agent` / `proprio` (agent xy) and `pos_target` (episode target xy); the code, the swm dataset keys
and TwoRoomEnv._set_state / _set_goal_state read `state` (2-d agent position) and `goal_state` (2-d target). This adds
the two columns as copies, once, and changes nothing else (the action NaNs at the last row of each episode come from
the recorder).

    python reproduce/data_prep/tworoom_add_state.py --h5 $DATA/tworoom.h5
"""
import argparse

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--h5", required=True)
a = p.parse_args()

with h5py.File(a.h5, "r+") as f:
    if "state" in f and "goal_state" in f:
        print("[add_state] state and goal_state present; nothing to do")
    else:
        pa = f["pos_agent"][:].astype(np.float32)
        pt = f["pos_target"][:].astype(np.float32)
        assert np.allclose(pa, f["proprio"][:].astype(np.float32)), "pos_agent != proprio"
        obs = f["observation"][:10000]
        assert np.allclose(obs[:, :2], pa[:10000]) and np.allclose(obs[:, 2:4], pt[:10000]), \
            "observation columns are not [agent xy, target xy]"
        if "state" not in f:
            f.create_dataset("state", data=pa)
        if "goal_state" not in f:
            f.create_dataset("goal_state", data=pt)
        print(f"[add_state] wrote state {pa.shape} = pos_agent, goal_state {pt.shape} = pos_target")
    print("[add_state] keys:", sorted(f.keys()))

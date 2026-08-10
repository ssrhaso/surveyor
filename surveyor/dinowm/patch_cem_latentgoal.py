"""Idempotent patch: teach CEMPlanner.plan to accept a pre-encoded latent goal
(drafted-subgoal serving). Flat path byte-identical.

The DINO-WM clone is located by $DINOWM_REPO (default ~/dino_wm)."""
import os
from pathlib import Path

p = Path(os.environ.get("DINOWM_REPO", os.path.expanduser("~/dino_wm"))) / "planning" / "cem.py"
src = p.read_text()
if "z_visual" in src:
    print("already patched")
else:
    old = """        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)"""
    new = """        if isinstance(obs_g, dict) and "z_visual" in obs_g:
            # pre-encoded latent goal (spec-accept serving): skip encode_obs
            z_obs_g = {"visual": obs_g["z_visual"].to(self.device),
                       "proprio": obs_g["z_proprio"].to(self.device)}
        else:
            trans_obs_g = move_to_device(
                self.preprocessor.transform_obs(obs_g), self.device
            )
            z_obs_g = self.wm.encode_obs(trans_obs_g)"""
    assert old in src, "cem.py drifted - patch target not found"
    p.write_text(src.replace(old, new, 1))
    print("patched cem.py")

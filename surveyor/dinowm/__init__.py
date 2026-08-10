"""DINO-WM transplant: the accept rule served inside a second architecture.

The scripts here run against an unmodified DINO-WM checkout: the CEM
sub-planner, world model and evaluator are theirs, and the only edit to their
tree is the guarded five-line latent-goal branch applied by
patch_cem_latentgoal.py. Locations are read from the environment so nothing is
tied to one machine:

    DINOWM_REPO   path to the dino_wm clone            (default ~/dino_wm)
    DINOWM_DATA   released checkpoints + encoded latents (default ~/data/dinowm)
    LEWM_DATA     root of the LeWM HDF5 datasets       (default ~/data)

Module map:

    planner.py                drafted-subgoal serving inside their MPCPlanner
    run_pusht.py              PushT planning runner, ARM=flat|spec
    encode_pusht.py           PushT latents through their own encode path
    encode_env.py             the same for any released env checkpoint
    encode_tworoom.py         Two-Room frames through raw frozen DINOv2
    patch_cem_latentgoal.py   idempotent latent-goal patch for their cem.py
    osf_fetch.py              list/download files from their OSF project
    patches/                  Two-Room env, dataset and configs for their tree
"""

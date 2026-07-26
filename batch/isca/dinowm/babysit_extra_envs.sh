#!/usr/bin/env bash
# Login-node chain: fetch point_maze + wall datasets from OSF, unzip, submit
# the encode+gap jobs. Runs under nohup; sbatch only fires after data exists.
set -e
cd /lustre/home/ha676/dino_wm
python3 osf_fetch.py get point_maze /lustre/home/ha676/data/dinowm
python3 osf_fetch.py get wall /lustre/home/ha676/data/dinowm
cd /lustre/home/ha676/data/dinowm/datasets
for z in *.zip; do
  [ "$z" = "pusht_noise.zip" ] && continue
  unzip -q -n "$z"
done
ls -d */ || true
cd /lustre/home/ha676/dino_wm
sbatch --export=ALL,MODEL_NAME=point_maze run_dinowm_envgap.sbatch
sbatch --export=ALL,MODEL_NAME=wall_single run_dinowm_envgap.sbatch
echo "BABYSITTER DONE"

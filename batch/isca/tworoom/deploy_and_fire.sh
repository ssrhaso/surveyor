#!/usr/bin/env bash
# One command to run the moment ISCA is reachable again. Pushes the current
# TwoRoom code (goal gate + cross-room split diagnostic) and submits the
# ceiling-and-gate battery, then reports what else is outstanding.
set -uo pipefail
cd /lustre/home/ha676/le-wm

echo "=== outstanding results to read ==="
for j in 2301412 2301459 2300262 2300331; do
  printf '%-10s ' "$j"
  sacct -j $j --format=State -X -n 2>/dev/null | sort | uniq -c | tr '\n' ' '
  echo
done

echo
echo "=== submitting ceiling + gate battery ==="
sbatch batch/isca/tworoom/run_tworoom_ceiling_and_gate.sbatch

echo
echo "=== flat with the cross-room split, both horizons, 2 seeds ==="
echo "(cheap: tells us whether flat's failures ARE the door-crossing episodes)"
H5=$(find /lustre/home/ha676/data/tworoom \( -name '*.h5' -o -name '*.hdf5' \) | head -1)
cat > /tmp/split_probe.sh <<EOF
cd /lustre/home/ha676/le-wm && source .venv/bin/activate
for OFF in 25 75; do
  RH=\$([ "\$OFF" -le 25 ] && echo 2 || echo 1)
  MODE=\$([ "\$OFF" -le 25 ] && echo short || echo long)
  python -m specaccept.envs.tworoom.eval --subgoal baseline \\
    --h5 "$H5" --source pretrained --encoder-id quentinll/lewm-tworooms \\
    --device cuda --mode "\$MODE" --goal-offset "\$OFF" --eval-budget \$((2*OFF)) \\
    --start final --eval-filter success --require-cross-room --episode-min 4000 \\
    --num-eval 64 --seed 42 --cem-seed 42 --receding-horizon "\$RH" \\
    2>&1 | grep -E "SR =|\\[split\\]"
done
EOF
sbatch --partition=gpu --gres=gpu:1 --account=research_project-deepmind \
  --nodes=1 --cpus-per-task=8 --mem=60G --time=02:00:00 \
  --job-name=2room-split \
  --output=/lustre/home/ha676/le-wm/logs/2room_split_%j.log \
  /tmp/split_probe.sh

echo
squeue -u ha676 -o "%.12i %.16j %.9T" | head -14

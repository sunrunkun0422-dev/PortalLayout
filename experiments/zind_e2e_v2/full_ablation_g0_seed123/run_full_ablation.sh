#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="/home/feixia/pythonProject/AAAsrk/Bi_Layout"
experiment_dir="$repo_root/experiments/zind_e2e_v2/full_ablation_g0_seed123"
checkpoint_root="$repo_root/checkpoints/End_To_End/zind_e2e_v2/g0_seed123"
log_file="$experiment_dir/training.log"
runs=(T0 T1 T2 T3)

cd "$repo_root"
exec >>"$log_file" 2>&1

printf '[%s] durable ablation runner started\n' "$(date --iso-8601=seconds)"
mapfile -t commands < "$experiment_dir/commands.txt"
if [[ ${#commands[@]} -ne 4 ]]; then
  printf '[%s] expected four commands, found %d\n' "$(date --iso-8601=seconds)" "${#commands[@]}"
  exit 2
fi

for index in "${!runs[@]}"; do
  run="${runs[$index]}"
  output_dir="$checkpoint_root/$run"
  completed_marker="$output_dir/.completed"
  if [[ -f "$completed_marker" ]]; then
    printf '[%s] %s already completed; skipping\n' "$(date --iso-8601=seconds)" "$run"
    continue
  fi

  command="${commands[$index]}"
  if [[ -f "$output_dir/last.pt" ]]; then
    command="$command --resume auto"
    printf '[%s] %s resuming from last.pt\n' "$(date --iso-8601=seconds)" "$run"
  else
    printf '[%s] %s starting from fixed independent initialization\n' "$(date --iso-8601=seconds)" "$run"
  fi

  printf '[%s] command: %s\n' "$(date --iso-8601=seconds)" "$command"
  bash -lc "cd '$repo_root' && $command"
  touch "$completed_marker"
  printf '[%s] %s completed successfully\n' "$(date --iso-8601=seconds)" "$run"
done

printf '[%s] all T0-T3 runs completed successfully\n' "$(date --iso-8601=seconds)"

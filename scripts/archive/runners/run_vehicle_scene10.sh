#!/usr/bin/env bash
set -euo pipefail

TF=/data_minio/data4/waymo/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training/training.tfrecord-00900-of-01000
RUN=/data_minio/data4/interhub/validation/vehicle_scene_10
PY=/root/miniconda3/envs/cy_interhub/bin/python
PROMPT=/data_minio/data4/interhub/prompt/qwen_generate_vehicle_descriptions_prompt.txt

mkdir -p "$RUN/vehicle_facts"
for i in $(seq 0 9); do
  "$PY" /data_minio/data4/interhub/scripts/facts/build_vehicle_facts.py \
    --tfrecord "$TF" \
    --record-index "$i" \
    --output "$RUN/vehicle_facts/scene_${i}.jsonl"
done

cat "$RUN"/vehicle_facts/scene_*.jsonl > "$RUN/all_vehicle_facts.jsonl"

"$PY" /data_minio/data4/interhub/scripts/llm/build_full_vehicle_facts.py \
  --input "$RUN/all_vehicle_facts.jsonl" \
  --output-dir "$RUN/full_vehicle_facts"

"$PY" /data_minio/data4/interhub/qwen_generate_vehicle_descriptions.py \
  --input "$RUN/full_vehicle_facts/full_vehicle_facts.jsonl" \
  --output-dir "$RUN/qwen_descriptions" \
  --prompt-file "$PROMPT" \
  --stationary-policy template \
  --workers 4 \
  --summary-csv "$RUN/qwen_descriptions/vehicle_qwen_descriptions_summary.csv"

echo RUN_COMPLETE

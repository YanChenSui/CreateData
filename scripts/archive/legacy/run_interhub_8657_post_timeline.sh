#!/usr/bin/env bash
set -euo pipefail

ROOT=/data_minio/data4/interhub/validation/interhub_8657_current
INTERHUB=/data_minio/data4/interhub
PY=/root/miniconda3/envs/cy_interhub/bin/python
TIMELINE_BATCH=$ROOT/02_pair_timelines/batch_results.json
TIMELINE_DIR=$ROOT/02_pair_timelines/pair_timelines
RAW_INPUT_DIR=$ROOT/03_llm_inputs/.raw_interaction_episodes
RAW_INPUT_JSONL=$ROOT/03_llm_inputs/.raw_llm_inputs.jsonl
FINAL_INPUT_DIR=$ROOT/03_llm_inputs/interaction_episodes
RAW_OUTPUT_DIR=$ROOT/04_llm_outputs/.raw_qwen_outputs
FINAL_OUTPUT_DIR=$ROOT/04_llm_outputs/descriptions
STAGE_LOG=$ROOT/06_reports/post_timeline_pipeline.log

mkdir -p "$ROOT/03_llm_inputs" "$FINAL_INPUT_DIR" "$ROOT/04_llm_outputs" "$FINAL_OUTPUT_DIR" "$ROOT/05_visual_text_validation" "$ROOT/06_reports"
exec > >(tee -a "$STAGE_LOG") 2>&1

echo "[WAIT] waiting for timeline batch"
while [[ ! -f "$TIMELINE_BATCH" ]]; do
  if ! pgrep -af 'run_interhub_pair_timeline_v3_batch.py' | grep -q "$ROOT"; then
    echo "[ERROR] timeline process ended before batch_results.json was written"
    exit 2
  fi
  sleep 30
done

echo "[STEP] extract raw interaction episode candidates"
rm -rf "$RAW_INPUT_DIR"
mkdir -p "$RAW_INPUT_DIR"
"$PY" "$INTERHUB/scripts/llm/extract_llm_inputs.py" \
  --input-dir "$TIMELINE_DIR" \
  --output-dir "$RAW_INPUT_DIR" \
  --output-jsonl "$RAW_INPUT_JSONL"
cp "$RAW_INPUT_DIR/failed_files.csv" "$ROOT/03_llm_inputs/failed_files.csv"

echo "[STEP] build InterHub visual mapping"
"$PY" "$ROOT/scripts/review/build_interhub_visual_mapping.py" \
  --manifest "$ROOT/00_manifest/interhub_records_manifest.jsonl" \
  --visual-dir "$INTERHUB/gif0/gif" \
  --visual-dir "$INTERHUB/gif/gif" \
  --visual-dir "$INTERHUB/gif1/gif" \
  --visual-dir "$INTERHUB/gif2/gif" \
  --visual-dir "$INTERHUB/gif3" \
  --output-csv "$ROOT/01_scene_mapping/visual_mapping.csv" \
  --output-jsonl "$ROOT/01_scene_mapping/visual_mapping.jsonl"

echo "[STEP] prepare stable interaction episode inputs and extraction audit"
rm -rf "$FINAL_INPUT_DIR"
mkdir -p "$FINAL_INPUT_DIR"
"$PY" "$ROOT/scripts/llm/prepare_interaction_episode_inputs.py" \
  --raw-input-dir "$RAW_INPUT_DIR" \
  --timeline-dir "$TIMELINE_DIR" \
  --final-input-dir "$FINAL_INPUT_DIR" \
  --output-jsonl "$ROOT/03_llm_inputs/llm_inputs.jsonl" \
  --audit-jsonl "$ROOT/03_llm_inputs/extraction_audit.jsonl" \
  --episode-map-csv "$ROOT/03_llm_inputs/episode_mapping.csv"

echo "[DISABLED] This archived pipeline used the retired interaction-description route."
echo "[DISABLED] Use qwen_generate_interaction_llm_descriptions.py with full_llm_facts_v1 instead."
exit 2

echo "[STEP] finalize descriptions, provenance, coverage, and visual-text records"
set +e
"$PY" "$ROOT/scripts/llm/finalize_interhub_descriptions.py" \
  --input-dir "$FINAL_INPUT_DIR" \
  --raw-output-dir "$RAW_OUTPUT_DIR" \
  --final-output-dir "$FINAL_OUTPUT_DIR" \
  --episode-map "$ROOT/03_llm_inputs/episode_mapping.csv" \
  --scene-map "$ROOT/01_scene_mapping/interhub_waymo_scene_mapping.csv" \
  --visual-map "$ROOT/01_scene_mapping/visual_mapping.jsonl" \
  --manifest "$ROOT/00_manifest/interhub_records_manifest.jsonl" \
  --timeline-batch "$TIMELINE_BATCH" \
  --output-jsonl "$ROOT/06_reports/final_pair_descriptions.jsonl" \
  --coverage-output "$ROOT/06_reports/coverage_report.json" \
  --visual-text-jsonl "$ROOT/05_visual_text_validation/visual_text_pairs.jsonl" \
  --visual-text-csv "$ROOT/05_visual_text_validation/visual_text_pairs.csv"
FINALIZER_RC=$?
set -e

echo "{\"qwen_exit_code\":$QWEN_RC,\"finalizer_exit_code\":$FINALIZER_RC}" > "$ROOT/06_reports/pipeline_exit_status.json"

if [[ "$QWEN_RC" -eq 0 && "$FINALIZER_RC" -eq 0 ]]; then
  echo "[CLEANUP] removing only generated staging directories"
  rm -rf "$RAW_INPUT_DIR" "$RAW_INPUT_JSONL" "$RAW_OUTPUT_DIR"
  find "$ROOT/scripts" -type d -name __pycache__ -prune -exec rm -rf {} +
  echo '{"removed":["03_llm_inputs/.raw_interaction_episodes","03_llm_inputs/.raw_llm_inputs.jsonl","04_llm_outputs/.raw_qwen_outputs","scripts/__pycache__"]}' > "$ROOT/06_reports/cleanup_manifest.json"
else
  echo "[KEEP] staging outputs retained because the run has failures"
fi

exit $(( QWEN_RC != 0 || FINALIZER_RC != 0 ))

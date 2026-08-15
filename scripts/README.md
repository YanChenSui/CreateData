# Pair interaction research pipeline

This directory contains the small reusable processing core.  The original
InterHub data-preparation chain remains at the project root and is
intentionally out of scope here:

- `0_data_unify.py`
- `1_interaction_extract.py`
- `2_case_visualize.py`
- `3_paper_plot.py`
- the original `utils/` data-processing implementation

## Active flow

The current entry-point layout is:

- `facts/`: pair/vehicle fact extraction and OOI candidate audit;
- `llm/`: the two fact projection/building steps;
- `archive/`: old experiments and compatibility helpers kept for reference;
- `archive/`: old experiments, compatibility helpers, semantic/export tools,
  and historical implementations kept for reference.

The description generation entry points are kept at the repository root:

- `qwen_generate_interaction_llm_descriptions.py`;
- `qwen_generate_vehicle_descriptions.py`;
- `deepseek_generate_vehicle_descriptions.py`.

Their prompt templates are under `prompt/`.

There is one active trajectory-processing method:

```text
InterHub classified record
        |
        | scene_id / scenario_index / agent_A / agent_B only
        v
scripts/facts/run_pair_facts_batch.py
        |
        v
scripts/facts/build_pair_timeline.py
        |
        v
pair_physical_facts_v1 JSON
```

`build_pair_timeline.py` reads the complete Waymo Motion Scenario protobuf and
emits physical facts only:

- physical lateral-maneuver evidence;
- conservative same/different travel-channel relation;
- longitudinal ahead/behind relation;
- own-vehicle acceleration/deceleration evidence.

InterHub windows are provenance fields only.  They do not crop trajectories,
select an event, or extend an event window.  This stage does not emit merge,
overtake, yielding, or other interaction semantics.

## Downstream stages

Description generation is downstream and must consume the facts output only
after the facts layer has been manually validated.  It is not invoked by the
active facts batch runner.

Older semantic, export, review, compatibility, and LLM-preparation tools are
under `scripts/archive/` and are not called by
`scripts/facts/run_pair_facts_batch.py`.

## OOI-only candidate audit (first stage)

`ooi_candidate_generator.py` is independent of InterHub.  It reads the raw
WOMD Scenario protobuf, uses `Scenario.objects_of_interest` as the candidate
group, filters to vehicle tracks, enumerates every unordered vehicle pair,
and optionally runs the existing facts-only builder for every pair:

```bash
python scripts/facts/ooi_candidate_generator.py \
  --tfrecord /path/to/training_splitted \
  --limit 10 \
  --output-dir /path/to/ooi_audit_10
```

It writes scenario-level OOI statistics, pair candidates, full
`pair_physical_facts_v1` JSONL results, and a separate error JSONL.  This
stage does not call the behavior classifier, semantic layer, or any LLM.

## Archived implementations

The earlier normalized `scene_motion_v3` timeline, its v2/v3 batch runners,
and the old raw-loader wrapper are under `scripts/legacy/`.  They are retained
for historical reproducibility and compatibility tests, but are not active
entry points.

One-off analysis, migration, retry, and debugging helpers that are not called
by the current entry points are under `scripts/archive/`.  They are retained
for reference but are outside the active pipeline and should not be used as
new execution entry points.

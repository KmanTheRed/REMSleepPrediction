# Blinded REM vision-language benchmark

This folder turns the four existing image datasets into a reproducible, blinded
benchmark without modifying any source file.

For the complete operator workflow, use [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md).

The frozen primary comparison is Claude Code with `claude-opus-5` at `max`
effort versus Codex/ChatGPT with GPT-5.6 Sol at Ultra effort/orchestration.

## What this fixes

- Public filenames are opaque and do not contain `pos`, `neg`, animal identity,
  or absolute time.
- Every original 30-minute target is re-rendered from raw signals with 5 minutes
  of real context before and after. Absolute time is removed, context is shaded,
  and the central 0--30 minute scoring region is labeled inside the image.
- Horizontal resolution per minute is preserved by widening the 40-minute image
  from 1960 to 2613 pixels instead of squeezing more time into the old width.
- Test bundles contain no labels, source manifests, earlier predictions, or
  scoring code.
- Calibration labels are separated from held-out inputs.
- One frozen evaluator applies the same deterministic postprocessing to the
  reference and prediction intervals: clip to `[0, 1800]`, merge overlapping
  intervals or gaps of at most 5 seconds, then discard merged intervals shorter
  than 30 seconds.
- Scoring includes per-second F1 and Cohen's kappa, one-to-one event matching
  over IoU 0.3--0.7, boundary error, REM-time bias, negative-frame false alarms,
  and fragmentation.

## Dataset roles

| Public stage | Source | Role | Labels exposed to model? |
|---|---|---|---|
| `calibration` | `qwen_rem_dataset` | 160 training + 40 development examples | Yes |
| `test_01` | `animal11_qwen_demo` | Held-out animal 1 | No |
| `test_02` | `newData/animal6_qwen_dataset` | Held-out animal 2 | No |
| `confirmatory` | `newData/animal5_qwen_dataset` | Locked confirmatory/reanalysis animal | No |

Animal 5 is **not historically sealed**: prediction artifacts already exist in
its source folder and the local Claude configuration records prior Animal 5
scoring work. It must not be described as a completely sealed prospective test.
A genuinely sealed claim requires a new animal whose labels and identity have
never been available to either system or the experimenter during prompt/model
development.

Animal 5 reference intervals are derived at build time from the latest
`manualscore5.txt`, using the required next-midnight clock rule. Its raw recording
starts at 20:51:38, so manual time zero maps to the next date at 00:00:00 and the
manual intervals receive an 11,302-second offset. Six legacy `labels_corrected`
files disagree with the newer manual file; they are retained only for the private
audit and are not used as benchmark truth.

## Build the blinded copies

Use the workspace environment or any Python 3.10+ environment with the packages
in `requirements.txt`:

```powershell
& '..\.venv\Scripts\python.exe' prepare_blinded_bundles.py
```

The command writes:

- `generated/model_inputs/`: safe-to-expose calibration and test material.
- `generated/private_vault/`: source mapping and all reference labels. Never put
  this directory in a model's working directory.
- `generated/SHA256SUMS.txt`: audit hashes for the frozen public inputs.

The generated folder is ignored by Git because it contains a private vault and
hundreds of rendered images. The ignored `private_id_salt.bin` preserves the same
opaque IDs and split across rebuilds. Use `--rebuild` only before any scored run;
the script refuses to replace an existing generated package otherwise.

Verify every public hash, image dimension, time origin, coverage exclusion, and
oracle score after a build:

```powershell
& '..\.venv\Scripts\python.exe' audit_bundles.py
```

## Run each system in a clean room

1. Freeze `prompt.txt`, the calibration set, all generation settings, and the
   evaluator before opening either held-out test.
2. Copy the calibration bank and exactly one public scoring stage to a new
   directory outside this repository:

   ```powershell
   python export_clean_run.py test_01 D:\vlm_clean_runs\system_test01
   ```

   For a scoring stage, the export contains all 200 calibration examples and
   labels plus only the images and opaque IDs for the selected held-out stage.
3. Start a brand-new model conversation/session in that directory. Do not attach
   this repository, a source dataset, a private manifest, a scorer, or a prior
   prediction file.
4. Give both systems the same `batch_task.txt`, calibration material, prompt,
   image resolution,
   per-image call policy, and opportunity to retry malformed JSON. Never show a
   score until the prompt is frozen.
5. Save one JSONL prediction file matching `predictions_template.jsonl`. Times
   remain seconds from the image's **0-minute score boundary**, not from the
   left edge of the 40-minute display.
6. Hash and register the run in `run_registry.csv`, then score it outside the
   model session.

For Claude Code, disable cross-session auto-memory and use a fresh directory.
The recommended noninteractive safety flags are documented in `PROTOCOL.md`.
For ChatGPT/Codex, use a fresh task with memory/reference-history features off;
the strongest reproducible OpenAI arm is a stateless API call with the exact
model snapshot or model ID recorded.

The identical high-level task for both agentic systems is in `batch_task.txt`.
It tells the system to study the calibration bank, score every held-out image,
and create a complete `predictions.jsonl` without seeing the evaluator.

## Score a run

```powershell
python evaluate.py `
  --vault generated/private_vault/frames.jsonl `
  --predictions path/to/predictions.jsonl `
  --stage test_01 `
  --output path/to/test_01_metrics.json
```

Use both `--stage test_01 --stage test_02` for the prespecified held-out aggregate
(with one combined prediction file containing all 400 expected frame IDs).
Keep `confirmatory` separate. The evaluator reports every animal/stage
separately and a micro-aggregate; do not bootstrap the 200 frames as though they
were independent animals.

## Before making the poster claim

- Run both systems once on `test_01` and `test_02` after freezing the protocol.
- Do not tune from either test score.
- Keep Animal 5 separate and call it confirmatory/reanalysis.
- Obtain at least two independent blinded human scorers plus an adjudicated
  consensus if the wording will say "matches expert performance." With one
  scorer, report agreement with that reference scorer instead.
- Describe this as a sampled 30-minute target-window benchmark with 5 minutes of
  visible context on each side. It is not directly
  comparable to a model evaluated continuously over a full recording unless the
  same sampling and postprocessing are applied to both.

Two source-edge exceptions are retained rather than fabricated or replaced:
one `test_02` frame has no available pre-context, and one `confirmatory` frame
has 3 of the requested 5 pre-context minutes. Their unavailable portions are
hatched and labeled `NO RECORDING` on the image. The source recordings also
contain four fully blank central windows in `confirmatory` and two central
windows below 90% coverage in `test_01`. All 200 images are still presented and
must receive predictions, but the evaluator excludes those six low-coverage
targets so missing-data negatives cannot inflate performance. The scored
denominators are therefore 200 calibration, 198 test_01, 200 test_02, and 196
confirmatory frames.

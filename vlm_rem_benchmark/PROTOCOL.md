# Frozen protocol

Version: 1.2 (model-configuration lock, frozen before model scoring)

## Objective

Compare two vision-language systems on REM interval localization using only the
information supplied in the current run: the frozen task prompt, the calibration
examples, and the image being scored. "No memory" means no user/chat history,
saved project instructions, cross-session model memory, retrieval, external
tools, or previous outputs. Reading the supplied images/files and writing the
prediction file are permitted and are not external knowledge. This control cannot
and does not remove knowledge encoded in pretrained model weights.

## Experimental design

The calibration animal supplies 200 labeled frames. A deterministic stratified
split assigns 160 to training and 40 to development. These are in-context/example
data, not model-weight training. Prompt changes are allowed only while looking at
calibration performance. Once calibration work is complete, freeze the prompt,
the policy by which the model may inspect the 200-example calibration bank, and
all settings. Then score `test_01` and `test_02` once per system. Do not return to
them for prompt selection. Analyze `confirmatory` separately.

The two model arms must receive the same pixels, labels, prompt, order policy,
and output repair policy. Record the exact product/interface, model ID, model
snapshot when available, reasoning/effort setting, image-detail setting, date,
and whether any tools were enabled. A "Claude Code versus ChatGPT/Codex" result
is an agent-system comparison, not a pure base-model comparison. For a pure VLM
comparison, add a stateless single-image API arm for each provider.

The primary agent-system configurations are frozen as:

- Claude Code with Claude Opus 5, pinned model ID `claude-opus-5`, at `max`
  effort.
- Codex/ChatGPT with GPT-5.6 Sol at Ultra effort/orchestration. Record the exact
  product and visible configuration. The corresponding OpenAI API model ID is
  `gpt-5.6-sol`, but Ultra is a Codex orchestration setting rather than the API
  `reasoning.effort` value. Do not relabel the agentic run as a stateless API
  call.

Freeze and record SHA-256 hashes for `prompt.txt`, `batch_task.txt`, the stage
bundle, and each prediction file. Use `run_registry.csv` so product names such as
"Ultra" are never substituted for the underlying model/interface details.

## Image and target geometry

Each image displays 40 minutes on an x-axis from -5 to 35 minutes. The middle
0--30 minutes are the only scored target. The -5--0 and 30--35 minute regions are
shaded context and are never themselves labeled or scored. White dashed lines at
0 and 30 minutes and a repeated in-image heading identify the scoring boundaries.
Predictions remain 0--1800 seconds relative to the 0-minute boundary, not the
left edge of the display. An event crossing a scoring boundary is clipped to 0
or 1800; an event lying entirely in context is not reported.

The plots are re-rendered from the same raw CBV and Motion Ang. Vector sources.
At 2613x700 pixels, the 40-minute plots preserve approximately the original
horizontal pixels per minute. No absolute timestamp is printed. One `test_02`
frame occurs at recording start and therefore has no real pre-context; one
`confirmatory` frame has only 3 minutes of real pre-context. The unavailable
regions are hatched and labeled rather than fabricated, extrapolated, or used to
replace a frame after labels were known.

All 200 images per stage remain in the model task. A target is eligible for
metrics only when at least 90% of its central 1,800 expected one-second samples
are present. This rule excludes two `test_01` targets and four `confirmatory`
targets, while excluding none from calibration or `test_02`. The model must still
return all 200 rows so hidden eligibility cannot influence its behavior. Missing
spans are broken rather than connected by a misleading line, and are hatched
`NO RECORDING`. Report the scored denominators (200/198/200/196) with results.

Manual-score clock alignment follows one frozen rule: if a recording starts at
00:00:00, that calendar date is the manual base date; otherwise the manual base
date is the next calendar date at 00:00:00. Thus Animal 5, whose recording starts
at 20:51:38, receives an 11,302-second offset. The current Animal 5 manual file is
newer than the legacy corrected JSON folder and disagrees with 6 of its 200
frames, so build-time intervals from the manual file and this clock rule are the
authoritative reference. The private audit records every disagreement.

## Blinding and contamination controls

- Work from only an exported clean-run directory containing `prompt.txt`, one
  public stage, and the empty predictions template.
- The model must not access the original repository, source filenames, manifests,
  label folders, private vault, evaluator, prior model outputs, or web.
- Use a fresh session for each system and stage. Do not continue a conversation
  that saw labels or scores.
- Process each held-out image independently. Do not feed prior held-out answers
  back into later image calls.
- Permit at most one deterministic JSON-format repair, using no score or label
  feedback. Log every repair.
- Randomize image order independently for each system or use the identical frozen
  order for both; choose before the run and record the choice.

Claude Code sessions are fresh by default, but project and user instructions,
plugins, saved transcripts, and auto-memory can still be loaded. Export outside
the repository and use a new `CLAUDE_CONFIG_DIR` for each scored run. For an
interactive subscription session, use:

```powershell
$env:CLAUDE_CONFIG_DIR = "D:\vlm_clean_runs\claude_config_test_01"
$env:CLAUDE_CODE_DISABLE_AUTO_MEMORY = "1"
$env:CLAUDE_CODE_SKIP_PROMPT_HISTORY = "1"
$env:CLAUDE_CODE_EFFORT_LEVEL = "max"
claude --tools "Read,Write" --no-chrome `
  --model "claude-opus-5" --effort "max"
```

Run interactively inside the exported clean directory. Approve only creation or
editing of `predictions.jsonl`; do not enable shell, web, or broader file access.
Authenticate inside that isolated configuration directory if prompted. Use a
different empty configuration directory for every system-stage run.

Do not add `--no-session-persistence` to this interactive command: Claude Code
documents that flag for print mode only. `CLAUDE_CODE_SKIP_PROMPT_HISTORY=1`
disables transcript and prompt-history writes in any mode. Bare mode is an
optional scripted/API-key route, not the default subscription route, because it
skips OAuth and keychain reads. If bare print mode is chosen, preregister it as a
different interface configuration and use it consistently for every Claude run.

Do not run Claude from this repository: its existing `.claude` state contains
prior scoring command history.

For a separate OpenAI API arm, use a new request for every image, `store: false`, no
tools, no `previous_response_id`, original/highest supported image detail, a
strict JSON schema, model ID `gpt-5.6-sol`, and a fixed API reasoning effort such
as `max`. Do not silently substitute the Codex Ultra orchestration mode for a
single-model call; label orchestration or multiple agents as an agentic system.

## Prediction format

One JSON object per line:

```json
{"frame_id":"F_0123456789abcdef","episodes":[{"start_sec":120,"end_sec":173,"confidence":0.82}]}
```

Every expected `frame_id` must occur exactly once and no unexpected ID may occur.
`episodes` may be empty. Start and end are seconds from the 0-minute boundary of
the central scored region. They are not seconds from the left edge of the
40-minute image. Confidence is optional and is not used for the primary score.

## Frozen postprocessing

Apply this independently within each central 30-minute target and identically to
the reference and predictions:

1. Clip endpoints to 0--1800 seconds.
2. Sort by start and end.
3. Merge overlaps and successive intervals whose gap is at most 5 seconds.
4. After merging, discard every interval shorter than 30 seconds.

Do not merge across frames. This order is normative: merge first, duration floor
second. The model reports raw candidate boundaries and must not perform the
5-second/30-second rule itself.

## Primary and secondary outcomes

Primary outcomes:

1. Per-second REM F1.
2. Per-second Cohen's kappa.
3. Event F1 using one-to-one within-frame matching at IoU 0.3.

Secondary outcomes:

- Event precision, recall, and F1 at IoU 0.4, 0.5, 0.6, and 0.7.
- Mean absolute onset and offset boundary error for pairs matched at IoU 0.3.
- REM-time bias: `(predicted_seconds - reference_seconds) / reference_seconds`.
- False-alarm rate among reference-negative frames.
- Fragmentation: mean number of predicted pieces overlapping a detected reference
  episode (ideal 1; report together with missed reference episodes).
- Malformed/repaired output count and inference cost/latency, if collected.

Report each animal/stage separately. The prespecified main held-out summary is the
micro-aggregate of `test_01` and `test_02`; Animal 5 remains separate. Frames and
episodes overlap in absolute time, especially in Animal 6, so frame-level or
event-level confidence intervals would be pseudoreplicated. Inferential claims
need additional animals; descriptive metrics are appropriate for this pilot.

## Human comparison language

Reference labels alone are not a human-performance distribution. "Matches or
beats experts" requires independent blinded scoring by multiple experts, an
adjudication rule fixed before evaluation, and human-versus-consensus metrics
computed with the identical sampling and postprocessing. Otherwise use:

> Agreement with the designated reference scorer was evaluated on held-out
> sampled frames.

## Stopping rule

Once a held-out stage has been scored, no prompt, example, threshold, parser,
model setting, or output-repair rule may change for that analysis. Any change
creates a new exploratory experiment and requires new held-out animals.

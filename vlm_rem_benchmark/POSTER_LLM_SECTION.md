# Poster-ready VLM methods text

## Vision-language REM localization

We evaluated whether vision-language systems could localize expert-annotated REM
episodes from paired CBV (IOS inverted) and motion plots. Each image showed a
central 30-minute scoring target plus 5 minutes of context on each side; only the
central target was labeled and scored. One animal provided 200 labeled
calibration frames (160 training, 40 development). After
freezing the task prompt, model settings, parser, and scoring rules, we evaluated
two held-out animals (200 sampled frames each). A third previously exposed animal
was analyzed separately as a locked confirmatory/reanalysis set, not as a sealed
prospective test.

For recordings beginning at a non-midnight clock time, manual-score time zero was
anchored to the following calendar date at 00:00:00. This added 11,302 s for the
confirmatory/reanalysis animal (recording start 20:51:38); reference intervals
were regenerated from the latest manual file before the frozen analysis.

All model inputs were blinded using opaque frame IDs and contained visible 0- and
30-minute scoring boundaries; source filenames, animal identity,
positive/negative indicators, absolute times, manifests, labels, prior
predictions, and scorers were withheld. Each system received identical pixels,
calibration data, batch instructions, and output schema in a fresh session with
cross-session memory, external tools, web access, and session persistence
disabled. Exact product interface, model ID, reasoning effort, and image-detail
setting were recorded before evaluation.

Models returned raw within-frame episode boundaries. Reference and predicted
intervals were clipped to 0--1800 s, overlapping intervals or gaps <=5 s were
merged, and merged intervals <30 s were discarded. Primary outcomes were
per-second F1, per-second Cohen's kappa, and one-to-one event F1 at IoU 0.3.
Secondary outcomes included event performance across IoU 0.3--0.7, onset/offset
error, REM-time bias, negative-frame false alarms, and fragmentation.
All 200 images per stage were presented; a frozen >=90% central-signal coverage
rule excluded 2 held-out-1 and 4 confirmatory/reanalysis frames from metrics.
Missing spans were visibly hatched, and scored denominators were reported.

## Results table template

| System | Animal | Second F1 | Kappa | Event F1 @ IoU 0.3 | Event F1 @ IoU 0.5 | Onset error (s) | Offset error (s) | REM-time bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Code / Claude Opus 5 / max effort | Held-out 1 | -- | -- | -- | -- | -- | -- | -- |
| Claude Code / Claude Opus 5 / max effort | Held-out 2 | -- | -- | -- | -- | -- | -- | -- |
| Codex/ChatGPT / GPT-5.6 Sol / Ultra | Held-out 1 | -- | -- | -- | -- | -- | -- | -- |
| Codex/ChatGPT / GPT-5.6 Sol / Ultra | Held-out 2 | -- | -- | -- | -- | -- | -- | -- |

Use the two-animal micro-aggregate only as a descriptive summary and show both
animal rows. Keep the confirmatory/reanalysis animal in a separate table or
panel. Because the pilot has only two independent held-out animals and sampled
frames overlap in absolute time, do not present frame-level confidence intervals
as though frames were independent biological replicates.

## Claim wording

Until multiple independent blinded human scorers are available, use:

> Vision-language systems were compared with a designated expert reference on
> held-out sampled frames.

Do not use "matched or beat expert performance" from comparison with one scorer.
If at least two independent experts and an adjudicated consensus are added, show
each human-versus-consensus result beside each model-versus-consensus result using
the identical sampling, postprocessing, and metrics.

# Step-by-step experiment guide

This guide is the operating procedure for the blinded comparison of Claude Code
and ChatGPT/Codex using GPT-5.6 Sol at Ultra effort. Follow it in order. Once a
held-out run begins, do not change the prompt, model settings, calibration bank,
image order policy, retry policy, evaluator, or postprocessing.

## 1. Decide exactly what the comparison means

The primary comparison is an **agent-system comparison**:

- Claude Code, with its normal image-reading workflow but no memory, web, shell,
  or files outside the clean run.
- ChatGPT/Codex with GPT-5.6 Sol and Ultra effort, with the same restrictions.

This is not a pure base-model comparison because the two products can inspect
files and organize work differently. If a pure VLM comparison is also desired,
add a separately named API arm with one stateless request per target image and a
fixed, identical calibration representation for both providers. Do not mix API
and agent-system results in one headline.

Use one primary run per system and stage. If repeat runs are desired, decide the
number before seeing any test result and report every replicate or their
prespecified mean. Never select the best run after looking at scores.

## 2. Freeze the experimental settings on paper

Before opening any held-out stage, record:

1. Claude Code with pinned model ID `claude-opus-5` and `max` effort.
2. Codex/ChatGPT with GPT-5.6 Sol and the visible Ultra setting.
3. Image-detail setting. Use the highest/original setting available.
4. Tool policy: local read and prediction-file write only.
5. Session policy: one new session for each system and each stage.
6. Order policy: use the frozen order in `frames.jsonl` for both systems.
7. Retry policy: at most one output-only repair, with no labels or score shown.
8. Primary metrics and the rule that Animal 5 remains separate.

Do not describe a UI label as an exact API model ID unless the product exposes
that ID. OpenAI documents `gpt-5.6-sol` as the GPT-5.6 Sol API model and supports
API reasoning efforts through `max`; Ultra is a Codex orchestration setting, so
the agentic run must still be recorded as the product configuration actually
shown in the interface.

## 3. Verify the frozen benchmark before every experiment batch

Open PowerShell and run:

```powershell
Set-Location 'D:\downloads\Kethan\Kethan\vlm_rem_benchmark'
& '..\.venv\Scripts\python.exe' audit_bundles.py
& '..\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Do not proceed unless the audit passes and all tests pass. The expected public
bundle hashes are:

| Stage | Expected SHA-256 |
|---|---|
| calibration | `957609cbdc824d31e2a42f961170ffb0e9a8697ff2370f40a18eb436b805970b` |
| test_01 | `ef41cc984ae5848076d9cf03fe37aec989d9ee3c611c5d38c25d006d73320cf8` |
| test_02 | `17548126ca85fb668f1c662382bd4cf0168ec06f4bd2c28bc67d3274463d7a43` |
| confirmatory | `9e43401e9e1c38fb618d5fd4ad15e479a0dc3c340ea1cee59799dca3658537fb` |

The frozen prompt hash is
`2D8116D44767D44EC27B8CB14B8C04743D4C09FFBA8F9702095FA4D7C724A3EC`.
The frozen batch-task hash is
`0DF906018F17A3FF7F3C5BEC9428086FFD9725AA83F95A40E8AA5E92D9E4BE75`.

If any hash differs, stop. Determine why before running either model. Do not
rebuild the bundles after a held-out prediction has been generated.

## 4. Understand what each model is allowed to see

Every clean scoring directory contains:

- 200 labeled calibration images: 160 train and 40 development.
- Exactly one unlabeled 200-image scoring stage.
- `prompt.txt`, `batch_task.txt`, and an empty prediction template.

It must not contain the original datasets, animal names, source timestamps,
private labels, `private_vault`, evaluation code, earlier predictions, prior
scores, web results, project instructions, or chat history.

The image itself spans -5 to 35 minutes. The shaded outer five minutes on each
side are context only. The model scores only the central 0-30 minutes and reports
times from 0 to 1,800 seconds relative to the left dashed scoring boundary. A
boundary-crossing event is clipped to 0 or 1,800 seconds.

The model reports raw candidates. It must not apply the duration/gap rules.
The evaluator later performs the frozen sequence on both predictions and truth:

1. Clip to 0-1,800 seconds.
2. Sort intervals.
3. Merge overlaps and gaps of 5 seconds or less.
4. Drop merged intervals shorter than 30 seconds.

## 5. Create six independent clean-run directories

Use a location outside `D:\downloads\Kethan\Kethan`. Never reuse a destination
or a model session.

```powershell
Set-Location 'D:\downloads\Kethan\Kethan\vlm_rem_benchmark'

& '..\.venv\Scripts\python.exe' export_clean_run.py test_01 'D:\vlm_clean_runs\claude_test_01'
& '..\.venv\Scripts\python.exe' export_clean_run.py test_02 'D:\vlm_clean_runs\claude_test_02'
& '..\.venv\Scripts\python.exe' export_clean_run.py test_01 'D:\vlm_clean_runs\chatgpt_sol_ultra_test_01'
& '..\.venv\Scripts\python.exe' export_clean_run.py test_02 'D:\vlm_clean_runs\chatgpt_sol_ultra_test_02'
```

Do not export or open `confirmatory` yet. Run the two main held-out stages first,
lock their predictions and analysis, and only then create:

```powershell
& '..\.venv\Scripts\python.exe' export_clean_run.py confirmatory 'D:\vlm_clean_runs\claude_confirmatory'
& '..\.venv\Scripts\python.exe' export_clean_run.py confirmatory 'D:\vlm_clean_runs\chatgpt_sol_ultra_confirmatory'
```

Additional directories are needed only for preregistered repeat runs. Give them
new names such as `claude_test_01_repeat_02`; never overwrite or continue a
prior run.

## 6. Run Claude Code on `test_01`

1. Close every previous Claude session.
2. Open PowerShell directly in the clean directory:

   ```powershell
   Set-Location 'D:\vlm_clean_runs\claude_test_01'
   $env:CLAUDE_CONFIG_DIR = 'D:\vlm_clean_runs\claude_config_test_01'
   $env:CLAUDE_CODE_DISABLE_AUTO_MEMORY = '1'
   $env:CLAUDE_CODE_SKIP_PROMPT_HISTORY = '1'
   $env:CLAUDE_CODE_EFFORT_LEVEL = 'max'
   claude --tools 'Read,Write' --no-chrome `
     --model 'claude-opus-5' --effort 'max'
   ```

3. Authenticate if prompted. The configuration directory must be new and empty
   before the run so user-level Claude instructions, plugins, and memory are not
   inherited. Use `claude_config_test_02` for `test_02` and another new directory
   for every later run.
4. Confirm that the session header shows Claude Opus 5 with max effort. Opus 5
   requires Claude Code 2.1.219 or later; update Claude Code before the experiment
   if the pinned model is unavailable.
5. Give this single instruction:

   ```text
   Read batch_task.txt and carry it out exactly. Your only deliverable is predictions.jsonl.
   ```

6. Allow reads only inside the clean directory. Allow writing only
   `predictions.jsonl`. Deny shell, browser, web, source-workspace, MCP, external
   retrieval, and broader filesystem requests.
7. Do not answer domain questions or give hints during the run.
8. Before ending, ask the same session only to verify valid JSONL, exactly 200
   unique expected IDs, and no extra IDs. This is format checking, not rescoring.
9. Close the session without showing it any labels, evaluator output, or scores.

Do not use `--no-session-persistence` interactively; it is a print-mode-only
flag. `CLAUDE_CODE_SKIP_PROMPT_HISTORY=1` is the documented control that applies
in any mode. Also do not add `--bare` to a subscription-authenticated run: bare
mode skips OAuth/keychain reads. A bare API-key run is permissible only if it is
preregistered as a separate interface and used consistently for all Claude
stages.

Repeat these steps in `D:\vlm_clean_runs\claude_test_02`, using a brand-new
Claude session and exactly the same settings.

## 7. Run ChatGPT/Codex Sol Ultra on `test_01`

1. Turn off memory and reference-chat-history features in the product settings.
2. Start a brand-new task whose only working folder is:

   `D:\vlm_clean_runs\chatgpt_sol_ultra_test_01`

3. Select GPT-5.6 Sol and Ultra effort. Record exactly what the interface shows,
   including the product name and date. Do not assume the UI configuration is
   identical to the API model merely because both say Sol.
4. Disable web and connectors. Do not attach the source repository or any prior
   chat. Do not let the task read parent folders.
5. Give the exact same one-line instruction used for Claude:

   ```text
   Read batch_task.txt and carry it out exactly. Your only deliverable is predictions.jsonl.
   ```

6. Permit only local reading and creation of `predictions.jsonl` in the clean
   directory. Do not provide mid-run hints.
7. Have that same task perform only the prescribed format check before it ends.
8. Close the task before any scoring occurs.

Repeat in `D:\vlm_clean_runs\chatgpt_sol_ultra_test_02` with a brand-new task.

If the product cannot guarantee that memory/reference history is disabled,
record that limitation and describe the result as a clean-folder, fresh-task
comparison rather than strictly memory-free. Pretrained model knowledge can
never be removed by this protocol.

## 8. Apply the identical failure and retry policy

For either system:

- A valid file has exactly 200 JSON objects, one per line.
- Every expected `frame_id` occurs exactly once; no unknown ID is allowed.
- `episodes` may be empty.
- Every interval has `start_sec < end_sec`, with numeric boundaries.
- Confidence is optional; if present it must be from 0 to 1.

If output is truncated or malformed, allow one repair in the same session with
only a format message such as: `Repair predictions.jsonl so it is valid JSONL
with all 200 expected IDs exactly once. Do not change already valid semantic
predictions.` Do not reveal which frames are positive, excluded, or incorrect.
Log the repair. If a second repair would be needed, mark the run failed rather
than silently giving one system more help.

## 9. Lock and hash predictions before evaluation

After all four main sessions are closed, create a results area and copy the
outputs into it:

```powershell
New-Item -ItemType Directory -Force -Path 'D:\vlm_results\claude' | Out-Null
New-Item -ItemType Directory -Force -Path 'D:\vlm_results\chatgpt_sol_ultra' | Out-Null

Copy-Item -LiteralPath 'D:\vlm_clean_runs\claude_test_01\predictions.jsonl' -Destination 'D:\vlm_results\claude\test_01.jsonl'
Copy-Item -LiteralPath 'D:\vlm_clean_runs\claude_test_02\predictions.jsonl' -Destination 'D:\vlm_results\claude\test_02.jsonl'
Copy-Item -LiteralPath 'D:\vlm_clean_runs\chatgpt_sol_ultra_test_01\predictions.jsonl' -Destination 'D:\vlm_results\chatgpt_sol_ultra\test_01.jsonl'
Copy-Item -LiteralPath 'D:\vlm_clean_runs\chatgpt_sol_ultra_test_02\predictions.jsonl' -Destination 'D:\vlm_results\chatgpt_sol_ultra\test_02.jsonl'

Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\vlm_results\claude\test_01.jsonl'
Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\vlm_results\claude\test_02.jsonl'
Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\vlm_results\chatgpt_sol_ultra\test_01.jsonl'
Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\vlm_results\chatgpt_sol_ultra\test_02.jsonl'
```

Enter one row per run in `run_registry.csv`. Record the prediction hash, exact
interface and model label/ID, reasoning setting, image detail, stage, UTC date,
memory status, tools, persistence, calibration access, repair count, and any
deviation. Never edit a prediction after hashing it.

## 10. Score each held-out animal separately

Run scoring only after its model session is closed:

```powershell
Set-Location 'D:\downloads\Kethan\Kethan\vlm_rem_benchmark'

& '..\.venv\Scripts\python.exe' evaluate.py `
  --vault 'generated\private_vault\frames.jsonl' `
  --predictions 'D:\vlm_results\claude\test_01.jsonl' `
  --stage test_01 `
  --output 'D:\vlm_results\claude\test_01_metrics.json'

& '..\.venv\Scripts\python.exe' evaluate.py `
  --vault 'generated\private_vault\frames.jsonl' `
  --predictions 'D:\vlm_results\claude\test_02.jsonl' `
  --stage test_02 `
  --output 'D:\vlm_results\claude\test_02_metrics.json'

& '..\.venv\Scripts\python.exe' evaluate.py `
  --vault 'generated\private_vault\frames.jsonl' `
  --predictions 'D:\vlm_results\chatgpt_sol_ultra\test_01.jsonl' `
  --stage test_01 `
  --output 'D:\vlm_results\chatgpt_sol_ultra\test_01_metrics.json'

& '..\.venv\Scripts\python.exe' evaluate.py `
  --vault 'generated\private_vault\frames.jsonl' `
  --predictions 'D:\vlm_results\chatgpt_sol_ultra\test_02.jsonl' `
  --stage test_02 `
  --output 'D:\vlm_results\chatgpt_sol_ultra\test_02_metrics.json'
```

The expected scored denominators are 198 for `test_01` and 200 for `test_02`.
Both models still had to predict all 200 images in each stage. The hidden
coverage filter excludes two low-coverage `test_01` targets without telling the
models which ones they are.

## 11. Produce the prespecified held-out aggregate

Create one combined prediction file per system without changing either source
file. In PowerShell 7:

```powershell
Get-Content -LiteralPath 'D:\vlm_results\claude\test_01.jsonl','D:\vlm_results\claude\test_02.jsonl' |
  Set-Content -LiteralPath 'D:\vlm_results\claude\heldout_combined.jsonl' -Encoding utf8

Get-Content -LiteralPath 'D:\vlm_results\chatgpt_sol_ultra\test_01.jsonl','D:\vlm_results\chatgpt_sol_ultra\test_02.jsonl' |
  Set-Content -LiteralPath 'D:\vlm_results\chatgpt_sol_ultra\heldout_combined.jsonl' -Encoding utf8
```

Then evaluate each micro-aggregate:

```powershell
Set-Location 'D:\downloads\Kethan\Kethan\vlm_rem_benchmark'

& '..\.venv\Scripts\python.exe' evaluate.py `
  --vault 'generated\private_vault\frames.jsonl' `
  --predictions 'D:\vlm_results\claude\heldout_combined.jsonl' `
  --stage test_01 --stage test_02 `
  --output 'D:\vlm_results\claude\heldout_combined_metrics.json'

& '..\.venv\Scripts\python.exe' evaluate.py `
  --vault 'generated\private_vault\frames.jsonl' `
  --predictions 'D:\vlm_results\chatgpt_sol_ultra\heldout_combined.jsonl' `
  --stage test_01 --stage test_02 `
  --output 'D:\vlm_results\chatgpt_sol_ultra\heldout_combined_metrics.json'
```

If Windows PowerShell writes a byte-order mark that the evaluator rejects, use
PowerShell 7 or combine the two files with a small binary-safe script. Do not
open and resave predictions in a spreadsheet or word processor.

## 12. Run Animal 5 only after the main result is locked

Animal 5 is a confirmatory **reanalysis**, not a historically sealed prospective
animal. Prior output artifacts and Claude project history already existed.

After the main test files, hashes, registry, and metrics are locked:

1. Export the two clean `confirmatory` directories described in Step 5.
2. Repeat the exact Claude and ChatGPT/Codex procedures with new sessions.
3. Hash and register both prediction files.
4. Evaluate each with `--stage confirmatory`.
5. Report the expected scored denominator of 196, separately from the main
   `test_01 + test_02` aggregate.

The evaluator's private Animal 5 truth is built from the latest manual score file.
Because its recording begins at 20:51:38 rather than 00:00:00, the manual base
date is the next calendar date at midnight, producing an 11,302-second offset.
The model never sees or uses this clock information.

## 13. Read the metrics in the prespecified order

Primary outcomes:

1. Per-second REM F1.
2. Per-second Cohen's kappa.
3. One-to-one event F1 at IoU 0.3.

Secondary outcomes:

- Event precision, recall, and F1 at IoU 0.4, 0.5, 0.6, and 0.7.
- Mean absolute onset and offset error for pairs matched at IoU 0.3.
- REM-time bias.
- False-alarm rate on reference-negative frames.
- Fragmentation and missed reference episodes.
- Output repair count, latency, and cost if collected consistently.

Report `test_01`, `test_02`, and their micro-aggregate. Keep `confirmatory`
separate. Do not bootstrap the overlapping frames as if they were independent
animals. With only two held-out animals, the comparison is descriptive rather
than a population-level inferential claim.

## 14. Use defensible poster language

If there is only one designated manual scorer, write:

> Agreement with the designated reference scorer was evaluated on held-out
> sampled 30-minute target windows, with five minutes of visible but unscored
> context on each side.

Do not write that the models matched or beat experts unless at least two
independent blinded experts score the same inputs and an adjudicated consensus
rule is fixed in advance. Human and model outputs must receive identical
clipping, merging, minimum-duration, missing-coverage, and metric rules.

The poster should also say that all 200 images were presented, predictions were
required for all 200, and hidden coverage exclusions produced scored
denominators of 198 and 200 for the two main held-out animals.

## 15. Final reproducibility checklist

Before treating the experiment as complete, verify that:

- The pre-run audit and six unit tests passed.
- Both systems received byte-identical public bundles and prompts.
- Every system-stage pair used a fresh session and clean directory.
- Memory/reference history, web, retrieval, and unrelated tools were disabled.
- The exact interface, model label/ID, effort, and image detail were recorded.
- Each prediction file has 200 unique expected IDs and was hashed before scoring.
- No score or hidden label was shown until that session was closed.
- The same evaluator and postprocessing were used for both systems.
- `test_01` and `test_02` are reported separately and as the frozen aggregate.
- Animal 5 is labeled confirmatory/reanalysis and reported separately.
- No model, prompt, threshold, parser, or repair rule was changed after seeing a
  held-out score.
- All deviations and failed runs remain in the registry; none are silently
  deleted or replaced.

Official product references:

- OpenAI GPT-5.6 Sol model: <https://developers.openai.com/api/docs/models/gpt-5.6-sol>
- OpenAI GPT-5.6 model guidance: <https://developers.openai.com/api/docs/guides/latest-model>
- Claude Opus 5 model details: <https://platform.claude.com/docs/en/about-claude/models/overview>
- Claude Code model and effort configuration: <https://code.claude.com/docs/en/model-config>
- Claude Code history isolation: <https://code.claude.com/docs/en/claude-directory>
- Claude Code CLI flags: <https://code.claude.com/docs/en/cli-reference>

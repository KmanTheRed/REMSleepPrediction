# REM event scoring with a duration-adaptive IoU gate

## Prespecified protocol

- Gate: `max(0.5, (d - 10)/(d + 10))`, where `d` is reference episode duration in seconds.
- Interpretation: IoU produced by an equal-duration prediction displaced by 10 seconds.
- The 10-second allowance is fixed before evaluation and is not fitted from prediction errors.
- Merge gaps of 5 seconds or less, then discard episodes shorter than 30 seconds.
- Apply the same postprocessing to reference and prediction intervals.
- TCN rule-label threshold: 0.80.
- TCN expert-label threshold: 0.50.
- Event matching: maximum-cardinality, ordered, one-to-one matching. Maximum summed IoU breaks ties.
- Aggregation for the headline values: unweighted macro-average across animals.

## Event F1 across complete recordings

| Cohort | Model | IoU 0.5 | IoU 0.7 | Duration-adaptive gate |
|---|---|---:|---:|---:|
| All six animals | Hand-tuned rules | 0.806 | 0.619 | 0.534 |
| All six animals | TCN, rule labels | 0.802 | 0.565 | 0.431 |
| All six animals | TCN, expert labels | 0.888 | 0.809 | 0.742 |
| Held-out animals A5, A6, A8, A11 | Hand-tuned rules | 0.833 | 0.622 | 0.552 |
| Held-out animals A5, A6, A8, A11 | TCN, rule labels | 0.816 | 0.549 | 0.401 |
| Held-out animals A5, A6, A8, A11 | TCN, expert labels | 0.899 | 0.807 | 0.747 |

## Held-out animals by caffeine condition

| Condition | Animals | Model | Duration-adaptive event F1 |
|---|---|---|---:|
| Pre-caffeine | A5, A6, A8, A11 | Hand-tuned rules | 0.559 |
| Pre-caffeine | A5, A6, A8, A11 | TCN, rule labels | 0.446 |
| Pre-caffeine | A5, A6, A8, A11 | TCN, expert labels | 0.775 |
| Caffeine | A5, A8, A11 | Hand-tuned rules | 0.613 |
| Caffeine | A5, A8, A11 | TCN, rule labels | 0.333 |
| Caffeine | A5, A8, A11 | TCN, expert labels | 0.668 |

A6 has no positive caffeine channel and therefore does not contribute to the caffeine subgroup. Fixed IoU 0.5 and 0.7 scores are shown as sensitivity analyses. The duration-adaptive gate becomes stricter than 0.7 for reference episodes longer than approximately 56.7 seconds, so its event F1 can be lower than the fixed 0.7 result.

These tables do not include Codex sampled-frame scoring because no continuous Codex event-interval stream was supplied for these six recordings.

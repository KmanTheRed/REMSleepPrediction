# Adaptive-IoU REM event evaluation

## Protocol

- Sources: the 18 reference and prediction files supplied on 2026-09-24.
- Common model coverage only.
- Half-open intervals: `[start, end)`.
- All reference and prediction streams: merge gaps of 5 seconds or less, then discard episodes shorter than 30 seconds.
- TCN trained on rule labels: `P(REM) >= 0.80`.
- TCN trained on expert/manual labels: `P(REM) >= 0.50`.
- Adaptive event gate: `IoU >= max(0, 1 - 20/L)`, where `L` is reference-event duration in seconds.
- Greedy one-to-one matching in descending IoU order.
- Headline summaries are unweighted macro-averages over animals.

## All six animals

| Detector | Adaptive precision | Adaptive recall | Adaptive F1 | F1 at IoU 0.5 | F1 at IoU 0.7 |
|---|---:|---:|---:|---:|---:|
| Hand-tuned rules | 0.563 | 0.603 | 0.580 | 0.806 | 0.619 |
| TCN, rule labels | 0.493 | 0.542 | 0.514 | 0.802 | 0.565 |
| TCN, expert labels | 0.780 | 0.760 | 0.769 | 0.888 | 0.809 |

## Held-out four animals (A5, A6, A8, A11)

| Detector | Adaptive precision | Adaptive recall | Adaptive F1 | F1 at IoU 0.5 | F1 at IoU 0.7 |
|---|---:|---:|---:|---:|---:|
| Hand-tuned rules | 0.595 | 0.600 | 0.596 | 0.833 | 0.622 |
| TCN, rule labels | 0.492 | 0.501 | 0.496 | 0.816 | 0.549 |
| TCN, expert labels | 0.787 | 0.759 | 0.772 | 0.899 | 0.807 |

## Adaptive F1 by animal

| Animal | Hand-tuned rules | TCN, rule labels | TCN, expert labels |
|---|---:|---:|---:|
| A1 | 0.473 | 0.602 | 0.707 |
| A3 | 0.623 | 0.499 | 0.821 |
| A5 | 0.731 | 0.510 | 0.760 |
| A6 | 0.330 | 0.245 | 0.776 |
| A8 | 0.703 | 0.654 | 0.813 |
| A11 | 0.621 | 0.574 | 0.739 |

The adaptive gate is stricter than IoU 0.7 whenever the reference episode is longer than 66.7 seconds. For example, the gate is 0.80 at 100 seconds and 0.90 at 200 seconds. This is why adaptive F1 is lower than fixed-IoU F1 for these data.

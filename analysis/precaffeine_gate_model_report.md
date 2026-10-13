# Pre-caffeine event F1 under alternative uncertainty models

All values are unweighted macro-averages across animals.

## Held-out animals A5, A6, A8, A11

| Detector | Constant e=10 s | Logarithmic | Linear | Hybrid |
|---|---:|---:|---:|---:|
| Hand-tuned rules | 0.559 | 0.618 | 0.606 | 0.581 |
| TCN, rule labels | 0.446 | 0.594 | 0.556 | 0.480 |
| TCN, expert labels | 0.775 | 0.816 | 0.808 | 0.790 |

## All six animals

| Detector | Constant e=10 s | Logarithmic | Linear | Hybrid |
|---|---:|---:|---:|---:|
| Hand-tuned rules | 0.530 | 0.597 | 0.585 | 0.557 |
| TCN, rule labels | 0.465 | 0.596 | 0.566 | 0.503 |
| TCN, expert labels | 0.764 | 0.815 | 0.808 | 0.789 |

The constant model is the primary analysis. The logarithmic, linear, and hybrid models are sensitivity analyses. All gates have a minimum IoU of 0.5. The logarithmic and linear models satisfy e(30)=10 seconds and e(150)=15 seconds. The hybrid model uses e(d)=max(10 seconds, 10% of duration), which has the same two anchor values.

# Scoring and validation

## Intended use

Sentinel prioritizes transactions for human review. It does not determine that fraud occurred, estimate a calibrated probability, block a payment, or make account decisions. Scores depend on the uploaded batch, the chosen currency, and the selected Isolation Forest sensitivity.

## Scoring pipeline

1. Validate the required columns, identifiers, timestamps, and amounts.
2. Sort activity by customer, timestamp, and transaction ID.
3. Calculate customer-history signals from prior rows only. The first five rows per customer are treated as history-building rows for amount, category, and country comparisons.
4. Fit an Isolation Forest to the uploaded batch using log amount, UTC hour, UTC weekday, category, country, and channel.
5. Add the rule points and 24 points when the forest marks a row as an outlier; cap the resulting score at 100.
6. Assign a band and flag rows at or above the user-selected review threshold.

The model is unsupervised: it is not trained from confirmed fraud labels. The expected outlier share slider sets the Isolation Forest contamination parameter. It changes how many observations the model labels as outliers; it is not the expected fraud rate.

## Signals and default points

| Signal | Points | Rule |
| --- | ---: | --- |
| Spending spike | 30 | Amount exceeds both two times the prior customer median and the median plus the larger of four median absolute deviations or 1.5 times the median. |
| New category | 14 | Category has not appeared in that customer's earlier history, after at least five prior rows. |
| New country | 22 | Country has not appeared in that customer's earlier history, after at least five prior rows. |
| Night activity | 12 | Timestamp is 23:00–23:59 UTC or 00:00–04:59 UTC. |
| Velocity burst | 25 | This row is part of a rolling ten-minute window containing at least three customer transactions. |
| High amount | 14 | Amount is at or above the 99th percentile of amounts in this upload. |
| Isolation Forest outlier | 24 | The model labels the row an outlier within this upload. |

Points can accumulate. Risk bands are Low (0–24), Medium (25–49), High (50–74), and Critical (75–100). The default review threshold is 50, so High and Critical rows are flagged by default. Changing the threshold changes the review flag, not the underlying score or band.

## Important limits

- Batch-relative: the forest and 99th-percentile cutoff are recalculated from each uploaded file. Results can change when the batch changes.
- Currency-sensitive: the app labels amounts but performs no currency conversion. Use one currency per file.
- History-sensitive: sparse or incomplete customer history can hide unusual behavior or create unstable baselines.
- No labels: unusual activity is not necessarily fraudulent, and ordinary-looking fraud may not be flagged.
- No streaming: the prototype analyzes uploaded records and does not monitor live transactions.
- No identity or device graph: it does not use card/device identifiers, IP addresses, geolocation distance, account age, or confirmed case outcomes.
- Review decisions are kept in the current Streamlit session and included in the CSV export; they are not written to a database.

## Validation before operational use

Use a representative, permissioned historical dataset with confirmed outcomes. Keep a chronological holdout period so future activity cannot leak into training or evaluation. At a minimum, review precision, recall, false-positive rate, alerts per 1,000 transactions, and analyst workload at the intended threshold. Compare results by relevant customer groups and transaction channels, inspect false negatives as well as false positives, and repeat the evaluation as behavior changes.

Tune the threshold and point weights on a validation period, then lock the choices before evaluating the holdout period. Recheck drift, data quality, and alert volumes after any changes. A qualified risk/compliance owner should approve any production use and human-review procedure.

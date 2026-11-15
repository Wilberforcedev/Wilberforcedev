# Data format and upload guide

Sentinel accepts a UTF-8 CSV containing one row per transaction. The uploaded file is analyzed as a batch; rows are not sent to an external fraud service by this app.

## Required columns

| Column | Type | Example | Notes |
| --- | --- | --- | --- |
| transaction_id | Text | TX-1007 | Required and unique within the file. |
| customer_id | Text | CUST-0042 | Stable identifier used to build customer history. |
| timestamp | Date/time | 2026-11-14T02:17:00Z | Pandas-parsable date/time. Signals use UTC. |
| amount | Number | 825000 | Must be positive. All rows must use the same currency. |
| merchant | Text | City Mart | Used for display and search. |
| category | Text | Groceries | Used in the model and the customer's history rules. |
| country | Text | Uganda | Used in the model and the customer's history rules. |
| channel | Text | POS | For example, POS, Online, Mobile, or ATM. |

Column names are case-sensitive and must match the names above. Extra columns are ignored. Blank merchant, category, country, or channel values become Unknown. Blank IDs, duplicate transaction IDs, invalid dates, missing/non-numeric amounts, and zero or negative amounts stop the upload with an error.

## Currency and time

Choose the currency used by every amount in the file from the sidebar. Sentinel adds that code to the display and export; it does not convert currencies or read an amount currency column. Do not mix currencies in one upload.

Timestamps with a timezone are converted to UTC. Timestamps without a timezone are treated as UTC. The night-activity rule and the exported timestamps also use UTC. Convert timestamps to the intended timezone before upload if your operational policy requires another reference timezone.

## Customer history

Rows are ordered by customer, timestamp, then transaction ID. Each row is compared only with earlier rows for that customer in the uploaded file. The first five transactions build history; historical amount/category/country signals start from the sixth transaction. A baseline median is shown when available.

For meaningful results, include enough recent historical activity per customer alongside the transactions you want to assess. The app does not persist an independent customer profile between uploads.

## Example

Use [sample_transactions.csv](../sample_transactions.csv) for a fictional example with prior Uganda activity and a later unusual transaction. The sample contains no real customer or account data.

# Design decisions

Each decision below gives the reasoning and what would change in production. These are the questions an interviewer is most likely to ask.

## Data engineering

**Why keep bronze as all strings (`inferColumnTypes=false`)?**
Bronze should be a faithful record of what arrived. If bronze casts `amount` to DOUBLE, a value like
`"12,50"` becomes NULL and the original is gone. Typing in silver with `try_cast` keeps the bad
value in bronze for debugging and sends the row to quarantine with a named reason.

**Quarantine vs. drop vs. fail: how do you choose?**
* *Fail* means the batch itself is broken and nothing downstream can be trusted (for example, a missing
  lineage column or a schema break). It is better to stop than to publish wrong numbers.
* *Drop to quarantine* means the row is unusable but the batch is fine (null amount, unknown merchant).
  Silver excludes it, and the quarantine table keeps it for triage and replay.
* *Warn* means the value is suspicious but legal (a $60k purchase). Keep the row and count the violation.
  Dropping it would bias the fraud data, because large purchases are where fraud concentrates.

**Why MERGE instead of append into silver?**
Upstream systems retry, which produces duplicate `transaction_id`s across files and across runs. MERGE
on the business key makes the job idempotent: re-running it changes nothing, and a corrected record
with a newer `_ingest_ts` replaces the old one.

**Why Auto Loader with `availableNow`?**
It gives incremental, exactly-once file discovery using checkpoints, run as a batch that processes the
backlog and stops. That suits serverless compute and a scheduled Workflow. Switching to continuous
streaming is a trigger change, not a rewrite.

## Machine learning

**Why an out-of-time split?**
Fraud patterns drift, and the features are themselves time windows. A random split puts transactions
from the same fraud burst in both train and test, which inflates the metrics. Training on the past and
testing on the most recent window matches how the model would actually be deployed.

**What is the label-delay issue?**
A transaction's fraud label usually arrives 30–90 days later, through chargebacks or customer reports.
The merchant fraud-rate feature therefore only counts labels at least 30 days old
(`rangeBetween(unboundedPreceding, -30d)`). Using same-day labels would let the model see the answer,
and it would fail in production. `sql/quality_checks.sql` includes a query that guards against this.

**Why PR-AUC and recall at 1% FPR instead of accuracy or ROC-AUC?**
With a fraud rate around 0.3%, predicting "never fraud" is 99.7% accurate. ROC-AUC is dominated by the
huge number of easy negatives. PR-AUC measures performance on the positive class, and recall at a fixed
FPR matches the real constraint: how much fraud we catch for a given number of customers we bother.

**How is class imbalance handled?**
Through class weights (LR) and `scale_pos_weight` (XGBoost). I did not use SMOTE, because synthetic
minority examples in a high-dimensional, time-dependent feature space tend to create unrealistic points.
Down-sampling legitimate transactions is available for train only (`legit_sample_frac`), so validation
and test keep the true base rate.

**How is the threshold chosen?**
By minimising expected cost on validation: `missed fraud $ + FP_cost × false alarms`, then applying that
fixed threshold to test. The $10 false-positive cost is an assumption covering customer friction, a
call-center contact and attrition risk. It is a parameter, and in practice it would be set with the
business. Weighting misses by transaction amount means the model is judged on dollars saved, not
just on counts.

**Why are the scores not calibrated probabilities?**
`scale_pos_weight` deliberately skews predicted probabilities upward. That is fine for ranking and
thresholding, and the calibration plot in 07 shows it. If downstream systems needed true probabilities
(for expected-loss reserving, say), I would add isotonic calibration fitted on validation.

**Why compare against logistic regression?**
It is a strong, interpretable baseline. If XGBoost doesn't clearly beat it on PR-AUC, the added
complexity isn't justified. In regulated credit settings, explainability also matters for adverse-action
reasons. That applies less to fraud, but the habit is good.

**What does the model miss?**
The `mimic` scenario: fraud that uses the customer's own merchants, amounts and device. Transaction-level
behavioural features cannot separate it from normal spending. Catching it needs other signals such as
login and session telemetry, device fingerprinting, or network and graph features linking accounts and
merchants.

**Why cluster customers?**
Segments make useful context. Thresholds could differ by segment, since a $900 electronics purchase is
normal for one segment and unusual for another. They also feed marketing and risk strategy. On synthetic
data, ARI against the hidden segment checks that the clusters reflect real structure rather than noise.

## What I'd do next in production

* Real-time scoring: Model Serving endpoint plus an online feature store for the velocity features
* Delayed-label handling for retraining (only train on transactions whose label window has closed)
* Drift monitoring per feature (PSI) and scheduled retraining triggered by a PSI or recall drop
* Champion/challenger deployment with shadow scoring before promotion
* Graph features (shared devices, IP addresses, merchant co-visitation) for organised fraud rings

# RSTGCN import and cold-start fit (Task 42 monitor step)

Run 2026-10-08 on the local dev DB (`msynd`), offline. Source zip sha256
`f8a6fe9e70972acb8ad98bbe5b7eb594c5b6b3475e21efd6d3b9e19049bdaf91` (verified by `import`).
Raw data and fitted model files stay on this PC (`data/models/` is gitignored).

## Import (`python -m msynd.ops.rstgcn import`, 40 s)

| Measure | Value |
|---|---|
| `obs.history_daily` rows | 1,282,325 |
| Runs (train, run_date) | 57,485 |
| Distinct trains | 3,892 |
| Run dates | 2024-09-01 … 2024-09-30 |
| Rows with occ > 1 | 0 |
| Rows with NULL seq | 0 |
| `ref.station` zones filled | 4,710 of 8,550 (none existed before) |

Trains by class (`ref.train`, UNKNOWN when not in GTFS): EXP 1,837; UNKNOWN 1,046; SF 554; SPL 172;
VB 126; SHT 62; DRNT 47; RAJ 26; MEX 20; PASS 2.

## Fit (`python -m msynd.ops.rstgcn fit`, 10 s)

Split: train = Sept 1–23, valid = Sept 24–30.

| Role | Train labels | Valid labels | Candidate log-score | Provisional log-score | Promoted |
|---|---|---|---|---|---|
| origin (O) | 43,975 | 13,510 | −1.872 | −3.641 | yes (ml.model 1) |
| run (R, composite) | 937,958 | 286,882 | −3.526 | −4.307 | yes (ml.model 2) |

Mean validation log-score in nats per label; higher is better. Both models are marked `cold_start`.
W (dwell) stays provisional. R is the composite (dwell + run) and so over-dispersed; live R replaces it
after 4 weeks (Task 38).

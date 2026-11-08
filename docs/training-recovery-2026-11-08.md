# Training recovery — 2026-11-08

Status: **BLOCKED on shadow-window maturity.** The cache repair and guarded
launcher are verified and installed. Full shadow scoring and weekly evaluation
have **not** started; neither evaluation report nor a promotion decision exists.
The weekly cron remains disabled, including the November 9 02:00 IST trigger.

## Containment and invariants

- Saved crontab, rotated weekly logs, container identity, registry, transactions,
  frozen manifest and image identity under the VM directory
  `/opt/msynd/deploy-backups/20261108-training-recovery/` (mode 700).
- Stopped `msynd-ops-run-ffffe80c2045`, started November 1 20:30 UTC, after
  rechecking its exact weekly command and image, with a 60-second allowance.
  Database transaction PID 1400609 closed. No candidate rows were committed.
- Active model IDs remain **1, 2**; frozen candidates **3, 4, 5** remain inactive.
  The complete registry is unchanged from the saved state. Benchmark loading
  successfully reloaded all candidate and baseline artifacts.
- All six artifact hashes match the frozen manifest, before and after work.
  The validation interval remains `[2026-10-22, 2026-11-05)` in IST, with
  **10,000 particles**, the frozen parameters and zero validation leg bias.
- No promotion, schema, model-format, collection-setting or scoring-rule change.

## Repair and proof

`tables_for` now retains the latest valid entry for each run/model-artifact set,
using the existing identity tuple. Version, feature-row, event, rescheduling and
bias checks are preserved, as is the 256-entry bound. Alternating candidate and
baseline models now retain table identity, allowing each forecaster to reuse its
replay across halts at the same issue time.

The three regression cases failed on the original code; the replay case observed
four replays instead of two. With the repair:

- Focused cache/baseline/weekly/shadow suite: **63 passed**.
- `uv run pytest -q`: **1001 passed, 1 skipped** (live test), one existing
  FastAPI/Starlette deprecation warning. Tests used isolated databases via
  `MSYND_TEST_DSN=postgresql://msynd:msynd@127.0.0.1:55432/postgres`;
  local `localhost` connections had stalled. Interrupted test databases removed.
- `npm --prefix web test`: **212 passed** across nine files.
- Shell syntax and `git diff --check` passed.

Retained original shadow image:
`sha256:01620019949a34f0cca66300dd8bd66f937566de51fdcdadcfd506e69571e607`.
Its cache module matches commit `ba77e91`.

Recovery image, tagged `msynd:training-recovery-20261108`:
`sha256:8a6e5e8c303abbeeceb41e761cdb932476191fd678df6db72ce1e84544e6c8fd`.
Its layers extend the original by one layer. A package-wide file-hash comparison
found exactly one changed file: `/app/msynd/engine/cache.py`, SHA-256
`ed60ad63e21846d0fae7e2a1217ecea3b6bc820748968fb5b8f0588d1e5b2bbe`.

The read-only benchmark used two different halts at one issue time for each of
14 deterministic runs, one per frozen validation date. Each run was selected by
descending observed-arrival support, then run ID. Both variants used the same
snapshot, frozen artifacts, seeds and full particle count; replay compilation
was warmed before timing. Complete forecast-array hashes and metrics match.

| Measurement | Original | Corrected |
|---|---:|---:|
| Cases | 28 | 28 |
| Inference calls | 168 | 84 |
| Replay calls | 56 | 28 |
| Elapsed seconds | 91.042 | 46.026 |

Measured speedup: **1.978×**. This measures the alternating-model defect; it does
not establish full-evaluation runtime or supply held-out screening results.
The first timed comparison exposed a NumPy JSON serialization error in the
benchmark report writer; that scratch script was corrected and the comparison
rerun successfully. Production code and scoring rules were unaffected.

## Guarded execution

`/opt/msynd/deploy/evaluate.sh` accepts `IMAGE weekly` or
`IMAGE shadow --model-id 3`. It resolves the image ID and shares
`/data/logs/evaluation.lock` across manual and scheduled evaluation, applies a
24-hour timeout with a 60-second termination allowance, and removes the named
container on exit. `/data/logs/evaluations.jsonl` records UTC start/finish, image,
job and exit status. The existing VM `shadow.sh score` also uses this launcher.

Real VM checks opened a transaction in the evaluation container and verified:

- Overlapping launch refused with exit **75**, including the legacy wrapper.
- Shortened timeout probe exited **124**; container and transaction disappeared.
- Launcher-only TERM initially reproduced deferred cleanup. After the fix to
  background the worker and use interruptible `wait`, it exited **143** and
  cleaned up in **0.37 seconds**. No evaluation containers or transactions remain.

The immutable-image weekly entry is installed **commented out**. Backups include
`crontab.guarded-disabled` and a prepared `crontab.guarded-ready`; do not install
the latter until both required evaluations succeed. Preserve unrelated cron
changes when restoring. Nightly cron and application services are unchanged.

## Blocking evidence and continuation

At November 8 15:04 UTC, the unchanged shadow pending predicate found **21** runs:
20 completed runs updated within six hours, and one nonterminal run:

| Run | Train | Start | State | Scheduled final arrival |
|---|---|---|---|---|
| 7995908 | 00290 | 2026-11-04 | scheduled, no event evidence | 2026-11-11 04:40 IST |

This is a seven-day timetable (final arrival offset 10,360 minutes), so the
frozen window has not fully matured. Existing expiry, if no actuals arrive,
occurs only after final arrival plus 12 hours; the shadow pending check also
requires six hours since the resulting run update. Recheck actual state rather
than assuming maturity at a fixed time. No lifecycle or evidence was fabricated.

Execution ruling: stop before the expensive full score because its unchanged
pending check already prevents `complete`; the benchmark proves scored cases
exist, so scoring now would report `partial`. Weekly evaluation and cron
restoration remain behind that gate. This defers partial-window metrics and the
weekly report until the completion prerequisite can pass.

After maturity, run the full shadow score under the installed launcher and save
its complete JSON. Require `complete`, support and confidence intervals; otherwise
record pending runs/missing evidence and remain blocked. Then run weekly
evaluation at its actual execution time with automatic week boundaries, verify
new candidates are inactive and reloadable, and restore Monday 02:00 IST cron.
If either corrected full evaluation hits 24 hours, verify cleanup and gather
profiling evidence before another retry.

Weekly promotion eligibility is **unassessed**: it still requires at least 2%
paired CRPS improvement, a positive CI lower bound, coverage within ±5 points,
and no class CRPS regression above 5%. Shadow results remain separate held-out
screening; promotion remains a separate decision.

VM evidence: `benchmark.json`, `benchmark-run.json`, `launcher-checks.json`,
`recovery-image.json`, `artifact-checks.json`, `recovery-status.json`,
`pending-runs.txt` and `pending-schedule.txt` in the saved recovery directory.

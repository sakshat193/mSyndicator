# Independent weekly evaluation implementation plan

> **For Luna** Use `superpowers:executing-plans` and execute this as one operational task in a fresh Luna chat. Use the checkboxes to track progress. Read `AGENTS.md` and the referenced implementation before execution.

**Goal** Complete one guarded weekly evaluation using the corrected cache, verify its inactive candidates, and report promotion eligibility while the separate frozen shadow window matures.

**Architecture** Reuse `deploy/evaluate.sh`, `msynd.ops.weekly.evaluate_week`, and the verified recovery image. A native transient systemd service owns the evaluation so an SSH disconnect cannot abandon its supervision. The existing launcher owns locking, the 24-hour limit, container cleanup, and execution events.

**Execution method** One Luna executor performs the task inline. This document is a handoff. Production execution begins when the user instructs Luna to execute it.

**Tech stack** Ubuntu 24.04, systemd, Docker, PostgreSQL 16, Python 3.12, psycopg, LightGBM, pytest, and the installed web test tooling.

**Spec** `docs/superpowers/specs/2026-10-07-live-train-tracker-design.md` section 8.6 and the implementation contract in `docs/superpowers/plans/2026-10-07-live-train-tracker.md`. Read `docs/training-recovery-2026-11-08.md` for the completed recovery evidence.

## Scope and ordering decision

Weekly evaluation is independent of the shadow scorer's pending-run check. This plan supersedes the November 8 execution ruling that deferred the manual weekly evaluation until shadow completion.

Cron restoration still requires both a successful weekly evaluation and a complete frozen shadow report. Model promotion requires a separate user decision.

The deliverable here is one verified weekly result. A rejected candidate is a valid completed evaluation. Insufficient support, empty scoring, timeout, or an unverifiable result requires a `BLOCKED:` report.

## Global constraints

- Treat this as task 104 if that task number is still unused. Recheck recent commits before starting and reuse an already completed equivalent attempt when its evidence satisfies this plan.
- Preserve active models, frozen artifacts, original images, model formats, schemas, receipt-time causality, scoring rules, and random seeds.
- Use the existing full 10,000-particle evaluation. The forecaster retains all N paths for scoring. Preserve the existing replay and publication-lag parameters.
- Keep candidates inactive through the existing `evaluate_week` path.
- Use the installed guarded launcher for the evaluation. Its host lock is `/data/logs/evaluation.lock`, its container is `msynd-evaluation`, and its limit is 24 hours with a 60-second termination allowance.
- Keep the weekly cron entry disabled throughout this task. Preserve nightly cron and application services.
- Keep the frozen shadow validation interval `[2026-10-22, 2026-11-05)` and candidates 3, 4, and 5 intact.
- Follow the repository's live-traffic restrictions. Use stored evidence and the launcher's existing offline configuration. Keep source evidence and model artifacts on the user's PC or VM.
- Use installed dependencies and platform tools. Runtime source changes require a separately reviewed repair if an unexpected defect prevents this operational task.
- Record operations and evidence on the VM. Copy only the aggregate report, verification summaries, and hashes into the local handoff evidence.
- Preserve the launcher's literal shell source and LF line endings.

## Known state to revalidate

These are November 8 observations, not a fresh production check.

| Item | Recorded value |
|---|---|
| Recovery commit | `fd8d975`, `task 103: recover bounded model evaluation` |
| Active model IDs | 1 and 2 |
| Frozen inactive IDs | 3, 4, and 5 |
| VM workspace | `/opt/msynd`, existing Ubuntu account with Docker access |
| Recovery directory | `/opt/msynd/deploy-backups/20261108-training-recovery/` |
| Recovery image tag | `msynd:training-recovery-20261108` |
| Immutable recovery image | `sha256:8a6e5e8c303abbeeceb41e761cdb932476191fd678df6db72ce1e84544e6c8fd` |
| Retained original image | `sha256:01620019949a34f0cca66300dd8bd66f937566de51fdcdadcfd506e69571e607` |
| Recovery cache SHA-256 | `ed60ad63e21846d0fae7e2a1217ecea3b6bc820748968fb5b8f0588d1e5b2bbe` |
| Launcher SHA-256 | `f4f33df67eb35ee7d2f23b43288927acd0c8d75ae656b605da08f1db8f932bc5` |
| Recorded verification | 63 focused tests, 1,001 full Python tests, 212 web tests passed |
| Benchmark | 28 cases across 14 dates, exact forecasts and metrics, 91.042 to 46.026 seconds |
| Blocking shadow run | 7995908, train 00290, start November 4, arrival scheduled November 11 at 04:40 IST |

The shadow run's current lifecycle must be checked separately. Its scheduled arrival is followed by the existing maturity rules. Its state does not decide whether this weekly task may start.

## Files and ownership

| Path | Responsibility in this task |
|---|---|
| `deploy/evaluate.sh` | Read and verify the installed launcher against the verified source |
| `msynd/ops/weekly.py` | Existing training, validation, report, registration, and rollback behavior |
| `msynd/eval/harness.py` | Existing paired-case promotion rules |
| `msynd/eval/baselines.py` | Existing frozen active-model forecaster and full-particle replay |
| `msynd/models/registry.py` | Existing artifact loader used for post-run verification |
| `tests/engine/test_cache.py`, `tests/eval/test_baselines.py`, `tests/eval/test_metrics.py`, `tests/ops/test_weekly.py`, `tests/ops/test_shadow.py` | Existing regression and gate verification |
| VM `/opt/msynd/deploy-backups/weekly-<UTC timestamp>/` | Private attempt evidence, stdout, stderr, registry snapshots, and report |
| `.codex-runs/weekly-evaluation-<UTC timestamp>/` | Local aggregate evidence copied from the VM |
| `docs/weekly-evaluation-2026-11-09.md` | New human-readable result, or use the actual execution date in the filename |
| `docs/training-recovery-2026-11-08.md` | Append a dated progress entry distinguishing the weekly result from the pending shadow result |

## Review focus

1. A pre-existing evaluation or lock conflict must lead to attachment or a documented blocker, without stopping another job. Stages 1 and 3 verify ownership and exit 75 handling.
2. IST week boundaries must follow the evaluator's real execution time. Stages 2 and 5 verify the report's `asof`, validation interval, and training cutoff.
3. Exit zero must be distinguished from useful evaluation completion. Stage 5 checks `fitted`, support, nonempty scores, eligible roles, and candidate rows.
4. Timeout, interruption, or lost stdout must be reconciled against database commits and artifacts. Stages 4 and 5 verify process cleanup and before/after identities before any retry.
5. Active-model or frozen-artifact drift must invalidate the claimed invariants. Stages 1 and 5 compare identities and hashes and reload every new candidate.

---

## Task 104 — Execute and verify one independent weekly evaluation

### Stage 1 — Reconcile current state and create attempt evidence

- [ ] Read the files in the ownership table. Inspect current Git status and recent commits. Preserve unrelated working files.
- [ ] Use the existing verified SSH connection from the recovery session. Verify the host key through the existing SSH configuration. Run Linux command blocks on the VM using a script over SSH stdin to avoid nested PowerShell quoting.
- [ ] Check `date -u`, host timezone, Docker access, `systemd-run`, `systemctl`, `flock`, `timeout`, and noninteractive sudo permission for a transient service. If the supervisor is unavailable, report the exact prerequisite as `BLOCKED:` before launching.
- [ ] Read current cron, evaluation event records, container identity, and relevant database sessions. Look for a current weekly attempt, including one started after the November 8 handoff.
- [ ] If an equivalent weekly attempt is running under the correct image and launcher, attach monitoring to it. If a completed equivalent attempt already covers the current validation weeks, verify and reuse its evidence. Investigate an unknown container or unexplained lock owner before starting another job.
- [ ] Confirm the weekly cron entry is disabled and the named evaluation container is absent when starting a new attempt. An unexpected enabled cron or orphan requires reconciliation before execution.
- [ ] Create one private attempt directory with a unique UTC timestamp. Record the chosen path and systemd unit name in the local progress notes.

Linux setup for a new attempt, after the ownership checks pass

```sh
umask 077
attempt="/opt/msynd/deploy-backups/weekly-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$attempt"
unit="msynd-weekly-$(date -u +%Y%m%d%H%M%S)"
image='sha256:8a6e5e8c303abbeeceb41e761cdb932476191fd678df6db72ce1e84544e6c8fd'
printf '%s\n' "$unit" > "$attempt/unit-name"
printf '%s\n' "$image" > "$attempt/image-id"
date -u +%FT%TZ > "$attempt/preflight-time"
crontab -l > "$attempt/crontab.before"
wc -c < /data/logs/evaluations.jsonl > "$attempt/events-start-offset"
```

- [ ] Save all model registry rows as JSON, including ID, role, kind, feature version, trained time, training interval bounds, path, active flag, and metrics. Save the active rows and their artifact hashes separately.
- [ ] Compare active IDs and frozen candidates with the recorded state. Reconcile a changed baseline or frozen manifest as an explicit blocker rather than silently selecting a replacement.
- [ ] Confirm both retained images exist. Resolve the recovery image ID and verify the cache digest inside it. Verify `/opt/msynd/deploy/evaluate.sh` against the recorded launcher digest and source. Save these checks.
- [ ] Load the saved frozen manifest from the recovery evidence and verify all six files against its hashes using a read-only model mount in the recovery image. Record checksums and success flags, keeping artifact contents private.
- [ ] Save available disk and memory figures and a baseline of running application services. Record storage problems or existing failures before adding load.

Useful registry snapshot query, executed through the existing `msynd-db-1` container

```sql
SELECT coalesce(jsonb_agg(to_jsonb(m) ORDER BY m.id), '[]'::jsonb)
FROM (
    SELECT id, role, kind, feature_ver, trained_at,
           lower(train_window) AS train_start,
           upper(train_window) AS train_end,
           path, active, metrics
    FROM ml.model
) m;
```

**Stage gate** The attempt has a verified image and launcher, saved baseline state, preserved frozen artifacts, and a clear execution owner.

### Stage 2 — Verify the existing evaluator and expected boundaries

- [ ] Run the focused regression suite against the isolated test database harness. Use the local development DSN with `127.0.0.1`, which resolved the earlier localhost startup stalls.

```powershell
$env:MSYND_TEST_DSN = 'postgresql://msynd:msynd@127.0.0.1:55432/postgres'
uv run pytest -q tests/engine/test_cache.py tests/eval/test_baselines.py tests/eval/test_metrics.py tests/ops/test_weekly.py tests/ops/test_shadow.py
uv run pytest -q
npm --prefix web test
```

- [ ] Confirm the local live-traffic guard before testing. Respect execution approval boundaries if the sandbox prevents access to the local test database or dependency cache.
- [ ] Save command exit codes and summaries. The prior full counts were 1,001 Python passes, one guarded live skip, and 212 web passes. Investigate discrepancies against current source rather than asserting those counts in advance.
- [ ] Verify these existing behaviors through the relevant tests and source.
  - `test_candidates_never_see_validation_rows` preserves training/validation separation.
  - `test_scores_on_identical_cases` preserves the paired comparison and decision.
  - `test_never_promotes` preserves inactivity for either acceptance outcome.
  - `test_failure_leaves_no_rows_or_files` covers Python-level rollback paths.
  - `test_insufficient_support_fits_nothing` prevents exit-zero support failures being mistaken for fitted candidates.
  - Promotion tests in `tests/eval/test_metrics.py` exercise CRPS gain, coverage, class regression, and case pairing.
- [ ] Compute the expected interval from the actual current IST week. For an execution whose `asof` falls on November 9, 2026, validation is `[2026-10-26, 2026-11-09)`, and training service dates precede October 26.
- [ ] Let the existing CLI choose `datetime.now(timezone.utc)`. The report's `asof` controls boundary verification if execution starts in a later week.

**Stage gate** Tests pass, and the expected weekly interval is recorded. The shadow pending predicate is outside this gate.

### Stage 3 — Launch durably through the existing guard

- [ ] Save a pre-launch manifest containing the unit name, attempt path, immutable image, Git revision, expected interval, active baseline IDs, launcher hash, and current UTC time.
- [ ] Start exactly one native transient service as the existing Docker-capable Ubuntu user. Keep its stdout and stderr in the attempt directory. The existing launcher supplies the timeout and shared lock.

```sh
sudo -n systemd-run \
  --unit="$unit" \
  --uid=ubuntu \
  --property=Type=exec \
  --property=UMask=0077 \
  --property=SupplementaryGroups=docker \
  --property="StandardOutput=append:$attempt/stdout.log" \
  --property="StandardError=append:$attempt/stderr.log" \
  --working-directory=/opt/msynd \
  /bin/sh /opt/msynd/deploy/evaluate.sh "$image" weekly
```

- [ ] Confirm the service starts, the launcher records a weekly start event with the correct image, and the container command is `python -m msynd.ops.weekly evaluate --models-dir /data/models`.
- [ ] Save the evaluation container ID, host start time, image, container IP, and matching PostgreSQL session identities. Record both backend PID and backend start time so later PID reuse cannot be confused with the original transaction.
- [ ] Interpret exit 75 as a lock or ownership conflict. Reinspect the legitimate owner and attach or report the conflict. A successful `systemd-run` submission alone does not prove the evaluation started.

**Stage gate** A supervised evaluation owns the lock, has a recorded identity, and uses the expected image and command.

### Stage 4 — Monitor through termination

- [ ] Check progress about every five minutes using service state, `docker stats --no-stream`, container state, stdout/stderr size, and the identified database sessions and wait events. Save timestamped snapshots on the VM.
- [ ] Treat a quiet stdout log or an idle database transaction during CPU replay as observations. The evaluator ordinarily prints its final JSON after fitting and scoring. Evaluate CPU-time deltas and database waits together before diagnosing a stall.
- [ ] Notify the user on meaningful progress, errors, termination, or required action. Maintain supervision until completion or an explicit blocker.
- [ ] On an SSH disconnect, reconnect and use the saved unit, container identity, attempt directory, and event-log offset to resume observation of the same attempt.
- [ ] Let the verified 24-hour guard enforce the runtime ceiling. Record timeout exit 124, interruption exit 143, other nonzero status, and unexpected termination distinctly.
- [ ] If an authorized early stop is necessary, stop only the identified transient service and verify the launcher's cleanup. Use saved identities to check the evaluation container and transaction have disappeared.
- [ ] On timeout or abnormal termination, preserve stdout/stderr, event records, CPU/memory samples, database wait/blocking information, container identity, and any available diagnostic output. Reconcile database rows and files before considering another attempt. Return `BLOCKED:` with that evidence before a retry. Identify any missing hot-path profiling and require a profiling-backed diagnosis before another full attempt.

**Stage gate** The attempt has a recorded finish outcome and cleanup proof, or a documented runtime/infrastructure blocker.

### Stage 5 — Validate the report and inactive candidates

- [ ] Preserve raw stdout and stderr. Parse the final nonempty stdout line as the weekly JSON object using stdlib JSON. Save a separate pretty-printed `report.json` and its SHA-256. Reject nonfinite numeric values or missing required sections in verification.
- [ ] Pair the weekly start and finish events after the saved log offset. Verify image identity, ordered timestamps, and exit zero. Resolve ambiguous or missing events before claiming success.
- [ ] Require `status == 'fitted'`, training support of at least 1,000 runs and seven service dates, both candidate and active score objects, positive and equal scored counts, and a structured `decision` with `accepted` and `reasons`.
- [ ] Treat `insufficient`, zero scored support, a missing final report, or inconsistent fields as a blocker even when the process exit is zero.
- [ ] Recompute the Monday boundary from the report's timezone-aware `asof`. Verify its two-week validation interval and training cutoff. Compare registry training bounds against the same interval.
- [ ] Read per-role support and floors. Every role with at least one `per_class` entry whose `below_floor` is false must have a reported candidate ID. Verify skipped roles against the saved support report. An attempt that fits no role does not satisfy the new-candidate deliverable.
- [ ] Capture the registry after execution. Require the newly committed model-ID set to equal the report's `model_ids`, and every reported candidate to have `active = false`, the expected role/kind, matching trained time, matching training bounds, and metrics consistent with the report.
- [ ] Reload every reported artifact through `msynd.models.registry.load_artifact(role, kind, path)` inside the verified image with a read-only model mount. Save role, kind, path, hash, and reload success. Verify required encoders load for a fitted run model.
- [ ] Compare active IDs and active artifact hashes with the pre-run snapshot. Verify the six frozen files and manifest again and confirm candidates 3, 4, and 5 retain their inactive state.
- [ ] Verify the acceptance metric at the report's `asof` agrees with `decision.accepted`. Verify the evaluation container and its original database session have closed and cron matches its pre-run state.
- [ ] If stdout was lost after a possible commit, inspect new registry rows and their persisted metrics first. Preserve evidence of committed candidates and orphan files. Report the missing-report condition as a blocker rather than silently refitting or reconstructing an allegedly complete original report.

**Stage gate** One complete weekly report is tied to a successful execution, reloadable inactive candidates, preserved baseline/frozen artifacts, and verified cleanup.

### Stage 6 — Report eligibility and record the handoff

The authoritative gate is `promotion_ok` as called by `evaluate_week`. Report these existing rules without changing their inputs.

| Gate | Required result |
|---|---|
| Pairing | Identical eligible case keys and matching outcomes/classes within the recent two full IST weeks |
| CRPS improvement | Candidate CRPS at most 0.98 times active CRPS |
| Confidence interval | Service-date block-bootstrap gain CI lower bound strictly greater than zero |
| Calibration | P10–P90 coverage within 0.75–0.85 and P05–P95 within 0.85–0.95, using the implementation's numerical tolerance |
| Class regression | Every evaluated class has candidate CRPS at most 1.05 times active CRPS, using the implementation's numerical tolerance |

- [ ] Save the existing `decision.accepted` and exact rejection reasons. An accepted decision establishes all implemented gates passed. For rejected decisions, map the reasons to the failed gates. An early pairing failure leaves later gates unassessed.
- [ ] Report overall candidate/active scores, coverage, support, role floors, fitted/skipped roles, model IDs, duration, image identity, and cleanup status.
- [ ] Identify the current reporting limit accurately. Weekly JSON contains overall summaries plus the gate decision and reasons. It omits the numeric CI endpoints and detailed per-class validation scores. Those values cannot be recovered from the aggregate JSON alone.
- [ ] Use the decision for promotion eligibility. Label unavailable numeric CI/class details as unavailable. The serialized overall score aggregates can include cases from service dates outside the promotion filter, so an overall percentage gain is informational rather than a replacement for the gate decision. Guard division by zero if displaying that percentage.
- [ ] Write `docs/weekly-evaluation-<actual date>.md` with the verified outcome and link the private attempt evidence. Append a dated independent-weekly update to `docs/training-recovery-2026-11-08.md`, retaining its historical sequence.
- [ ] State shadow status separately, using a fresh read-only lifecycle/maturity check if needed. Record that cron restoration still awaits both required evaluations and that promotion is a separate decision.
- [ ] Review the document diff and commit only this task's result documentation as `task 104: evaluate weekly candidate independently`, provided task 104 is still unused. Keep raw attempt output and scratch verification under the private VM directory or gitignored local evidence directory.

**Final response** Give the weekly outcome, accepted/rejected eligibility and reasons, support, inactive candidate IDs, elapsed time, proof of unchanged active IDs and frozen hashes, and the report path. On failure, start with `BLOCKED:` and give the exact blocker, cleanup state, and evidence path.

## Completion checklist

- [ ] One equivalent current-week evaluation completed with exit zero and a valid fitted report.
- [ ] Candidates were registered inactive, matched the report, and reloaded successfully.
- [ ] Promotion eligibility and reporting limitations were stated accurately.
- [ ] Baseline model identities and frozen shadow hashes matched their saved state.
- [ ] Evaluation container and original transaction closed, with recorded start/finish/image/status.
- [ ] Weekly cron remained disabled pending the separate recovery exit gate.
- [ ] Results and the remaining shadow prerequisite were documented and handed back to the user.

## Planning verification and limits

This plan was checked against the local evaluator, gate implementation, launcher, regression tests, task 103 evidence, and current local hashes on November 9. Production state and systemd availability must be revalidated by Luna in Stage 1. The 1.978× benchmark establishes the cache improvement on its 28 cases. Full weekly runtime and the eventual promotion outcome require the actual evaluation.

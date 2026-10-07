# Live train tracker: v2.1 follow-up review

**Verdict: architecture accepted as a basis; not approved for implementation planning yet.** Three correctness blockers remain in the written contracts. Phase 0 and retention also need the narrow corrections below. No product implementation or implementation plan was written, and the spec was not edited.

Reviewed the complete [spec](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md), concentrating on sections 3, 4 and 15. Snapshot: commit `9042cfa`, spec SHA-256 `aae1b04eae6cf5fbf5c791bf4aa0109a2460d6d2665ce85c11a3b626c997f73a`.

**Proof:** `python docs/superpowers/reviews/2026-10-07-live-train-tracker-v2.1-checks.py` passed 11 focused design checks on Python 3.10.11. These are reference calculations and counterexamples to specified rules, not tests of an existing engine. No Oracle benchmark, live pilot, or PostgreSQL integration test was performed. [Runnable checks](B:/m-syndicator/docs/superpowers/reviews/2026-10-07-live-train-tracker-v2.1-checks.py).

## What the previous fixes resolve

| Previous finding | v2.1 assessment |
|---|---|
| Downstream-only conditioning left the upstream posterior at 1/2 | Weighting and retaining ancestors resolves the 2/3 counterexample. The general proposal is sound when its conditional draws and likelihoods agree. The rare-window fallback below breaks that agreement. |
| Pending meant not happened | The soft survival likelihood addresses this. The pilot must not turn a finite zero-failure sample into a hard guarantee. |
| Filtering current truth lost earlier revisions | The new SQL filters before selecting the latest eligible revision, with a deterministic tie-break. That query addresses the defect. ETA confirmation remains too broad. |
| Rescheduling counted twice | The residual definition and initialization resolve scheduled 100, rescheduled 20, actual 130. Its distribution support was not updated to match the new label. |
| Ten-second frames contradicted A6 | Immediate deltas remove that cadence contradiction. Actual latency remains an empirical G8 result. |

G7 before performance work, history quarantine, optional Kaggle, and profiling before a Rust port are appropriate. Pending empirical gates do not themselves prevent planning when their contracts and failure paths are coherent.

## Three correctness blockers

### 1. [P1] Rare positive-probability windows still use the wrong proposal

**References:** [truncation fallback, line 231](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:231); [importance weight, line 259](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:259); [consistency claim, line 274](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:274).

Give the actual interval [0,1) prior mass 5e-7, uniformly distributed within that interval, and set tau=1. The correct conditional draw is uniform, with mean 0.5 and no mass at 1. The specified fallback draws min(Exp(1),1): mean 0.632121, with 36.79% of draws exactly at the excluded upper endpoint. Every particle can have the same positive weight 5e-7, so resampling cannot correct this wrong conditional distribution. Increasing N does not remove the bias.

**Blast radius:** observed-time jitter, bridge reconstruction, downstream delay features and forecasts. Rare evidence is precisely where reliable conditioning matters. The asymptotic-posterior claim is false with this fallback.

**Fix:** for every positive window probability, sample the actual conditional distribution and use its matching likelihood. A zero-mass particle gets zero weight; retain the existing flagged override only when the complete proposal cloud is incompatible. Use stable probability calculations rather than classifying every mass below 1e-6 as impossible.

Also define F as the **effective event-time distribution**, including the departure-time shift, delay interpolation, bias and physical floor. A floor creates atoms: for D uniform on [0,10], max(5,D) lands in [5,6) with probability 0.6, while the raw CDF gives 0.1. Specify half-open interval probabilities correctly at atoms. Inverse-quantile interpolation is not a mixture of the neighboring CDFs.

**Required proof:** add a rare but feasible window, a floor atom at an interval boundary, and a mixed feasible/infeasible ancestor case to G7. All surviving draws must be inside the actual interval and match its conditional law.

### 2. [P1] O is signed, but its output format remains nonnegative

**References:** [residual definition, line 179](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:179); [bins, line 222](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:222); [neural head, line 669](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:669).

Scheduled departure 100, latest rescheduling +60, actual departure 150 gives O=-10. Actual departure is still 50 minutes late against the original timetable. The contract does not forbid this observation; its exception selection uses receipt time, which also does not establish that the train departed after the selected rescheduled time. Nevertheless, O's 0..180 bins can only initialize departures at 160 or later.

**Blast radius:** origin labels, empirical tables, LightGBM rasterization, neural output dimensions, initial forecasts and spurious data-invariant flags on otherwise permitted observations.

**Fix:** give O signed support and consistent tail semantics, updating all consumers. Reuse a suitable existing signed format where possible. Alternatively, an authoritative source guarantee would have to justify and explicitly enforce a different label domain; that guarantee is currently absent. Keep W nonnegative.

**Required proof:** preserve the +20 rescheduling regression and add this negative-residual case through label construction, distribution conversion, sampling and initialization.

### 3. [P1] A successful run fetch can falsely refresh an absent ETA

**References:** [baseline contract, line 203](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:203); [fetch confirmation, line 437](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:437); [ETA storage, line 470](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:470).

An ETA for halt B is seen at 09:00. An otherwise valid 10:00 status response contains another halt's ETA but omits B. At 10:10, the specified run-wide confirmation makes B's stored ETA eligible as 10 minutes old. B was actually last confirmed 70 minutes ago. Neither parser validation nor the source contract requires every successful response to contain every previously stored ETA.

**Blast radius:** B2 baseline eligibility, identical-case comparisons, A3 acceptance, weekly promotion, and displayed ETA ages. A stale baseline can make the learned model appear better; selective eligibility can also change which cases are scored.

**Fix:** freshness belongs to the specific (run, station, occurrence, kind, value) actually parsed. The simplest existing-table solution is to append each present ETA on every successful response, including unchanged values, and use its own receipt time. Missing keys get no new confirmation. Measure the extra rows under G5/Phase 2 rather than retaining the old storage estimate unquestioned.

**Required proof:** test unchanged-present, missing, changed, and post-cutoff confirmations separately. A successful token/session request must never confirm an ETA.

## Phase 0 and operational corrections

### 4. [P2] G2 needs a feasible cohort and a defensible tail rule

**Reference:** [G2, line 1016](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:1016).

Forty **simultaneous** runs at two HTTP requests per enquiry every 120 seconds require 0.667 requests/s, above the pilot's 0.3 limit. At most 18 fit even before boards, exceptions, sessions and retries. Forty distinct runs over the week can be feasible if cohorts rotate; the current wording does not specify this.

**Fix:** explicitly distinguish the total sample from the simultaneous cohort. Bound the latter using measured h and reserve capacity for other jobs. Keep the agreed rate.

The wording making a source with zero counterexamples a candidate for a hard bound needs tightening. A hundred observations miss a real 1% tail with probability 36.6%. A finite pilot can estimate uncertainty, not establish a timestamp guarantee. An event missing at the end of the pilot is also right-censored; disappearance from a board is not proof that it was never published.

**Fix:** require an independently verified source guarantee for any hard bound, as section 3 already requires. Otherwise retain a soft tail, state the uncertainty in epsilon, and test sensitivity to long-lag and unobserved-event assumptions.

**Blast radius:** an infeasible pilot cannot validate the publication model; an unjustified zero tail can again eliminate the true trajectory for every affected run.

### 5. [P2] Fetch retention conflicts with permanent evidence foreign keys

**References:** [event fetch FK, line 451](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:451); [history fetch FK, line 485](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:485); [retention, line 926](B:/m-syndicator/docs/superpowers/specs/2026-10-07-live-train-tracker-design.md:926).

Fetch rows expire at 180 days, but actual revisions and history rows continue referencing them. PostgreSQL's default FK deletion behavior rejects deletion of a referenced parent. Deleting expiring pending/ETA rows first does not remove the permanent event/history references.

**Blast radius:** retention failures, growing request-log storage, failed nightly work, and eventually disk exhaustion. Cascading deletion would instead destroy retained training evidence.

**Fix:** explicitly preserve small fetch provenance indefinitely or make retained child references nullable with ON DELETE SET NULL. Choose one policy and apply it consistently; never cascade into evidence. Add an old referenced-fetch retention check. The companion script demonstrates the corresponding default NO ACTION behavior using SQLite; the actual Postgres migration still needs its own check.

## Narrow additions to G7 and the implementation hand-off

- **Repeated pending:** for one persistent publication delay, later non-report subsumes earlier non-report. Use the latest cumulative marker per event/source, or a joint observation likelihood, rather than multiplying marginal survival probabilities. In the companion example, multiplication changes a posterior from 0.731059 to 0.880797. Make actual-versus-pending selection explicit. The current latest-revision wording suggests deduplication but does not fully specify the merge across tables.
- **Pass approximation:** define its prior and report reference-model error against a declared acceptance criterion. Merely reporting an arbitrary error cannot decide whether this approximation is usable. Separate simulation error from approximation error.
- **Monte Carlo tolerance:** estimate error across reproducible independent replicates; resampled descendants are correlated. Compare discrete marginal probabilities/CDFs as well as quantiles. An iid binomial standard error is not automatically valid after repeated resampling.
- **Dirty batches:** the displayed DELETE consumes every dirty row despite the stated cap of 200. Specify a bounded consume and acknowledgment after successful output publication; failed work must be retried. Startup's full rebuild covers a process crash, so this review does not claim permanent loss solely from a crash between delete and replay.
- **Latency endpoint:** clarify whether A6 stops at the database frame, API observation, or client delivery. Engine fallback is 1 second; API fallback is 5 seconds. Test notification loss against the chosen endpoint, alongside normal-notification latency.

No redesign is needed. Resolve the three correctness contracts, make G2 executable without inventing a hard guarantee, and reconcile retention before the implementation plan. Source feasibility and Oracle performance remain gate results to be measured, not prerequisites to assert without evidence.

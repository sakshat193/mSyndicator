# Overnight run 2026-10-09: things for you to check

Each item is a reversible ruling I made as monitor, so work could continue. Overrule any of them and I'll redo it.

1. **Task 24 engine entrypoint.** The plan never gives the engine a `main()` (`python -m msynd.engine` exits doing nothing).
   I let Task 24 add a minimal `main()` (DB URL from settings, `run_loop` with defaults) and one wiring test.
2. **Task 29 disruption days.** The spec defers storing network surprise counts, but A4 breakdowns need "top-decile
   disruption days". Ruling: the harness derives them from the PITs of the scored cases (surprise = PIT < 0.001 or
   > 0.999, counted per IST service day, top decile). An optional explicit `disruption_days` set overrides it.
   Marked `# ponytail:` (upgrade: stored counts once the engine persists them).
3. **Task 22 under MMT.** MMT has no boards and no exceptions page, so those admission slots are left out (not stubbed).
   T_max, F and the board share are parameters with provisional defaults; they are CP-3 decisions.
4. **Frontend (optional).** Plan only, nothing built: `reports/frontend-live-plan.md`. It needs your decisions A–D.
5. **Task 22 merged (Codex's ambiguities).** Breaker recovery is a plain MMT status probe (MMT has no session).
   T_max defaults to `1.25·R·h/rate`; `--t-max-seconds` overrides it. F stays at `F_MINUTES=60`. The etrain backfill
   flag is refused outright until G1/G4. All of these are CP-3 decisions. I tightened the `MSYND_LIVE` refusal test,
   because a guard-removal mutation had survived (`real_transport` caught it downstream). The collector cannot start
   while the pilot holds the MMT host lock.
6. **Task 24 merged.** Codex wrote the files, then hit a timeout and its usage limit; I ran the verification myself.
   The suite is green (388 on the merged branch). `docker buildx build --platform linux/arm64,linux/amd64` succeeded,
   and `import lightgbm, onnxruntime, numba` works in both arches. I ran it with `docker run` on the built image,
   not via `compose run ops`, because that service bind-mounts host `/data`. The compose collector defaults to
   `MSYND_LIVE=0`, and the app services sit under profile `app`, so `docker compose up -d db` is unchanged.
7. **Codex usage limit** until 06:57. Task 29 (evaluation harness) is half-written in `.worktrees/task-29`; a resume
   was queued for 07:05, but Claude Code stopped that waiter because the PC was low on memory. It is
   unreviewed and unmerged; I have not restarted it. To continue: `codex exec resume --last` from that worktree.
8. **Task 29 merged (09 Oct, morning).** Codex gpt-6.1-sol fixed B3 freezing and early-departure cuts. Its ambiguities:
   B2 reads MMT ETAs from the legacy-named `obs.ntes_eta_rev` table; B3 uses zero leg bias (leg_bias has no receipt
   timestamp, so a historical forecast could otherwise see later nightly corrections); `n_curve` needs an explicit
   forecaster. Mutation checks killed 8 of 9 real mutations; I added a low-tail (PIT < 0.001) disruption-day assertion
   for the survivor. The Phase 3 exit box stays open (it needs 4 weeks of live data).

## 9. Dashboard build (2026-10-09, user decisions A–D approved)
- Plan amended (b229e5c): F1 pilot bridge; Task 36 part 1 = minimal map, no basemap; live = pilot cohort only.
- F1 merged (78c0f4d). F1b (c5022c3): engine spawn pool could not import `_job` under `python -m` — Docker engine was broken too; fixed.
- User ruling, supersedes Task 24 item: engine main() loads S from data/params.json (provisional now; G2 fit replaces it). Task F1c.
- Memory: run one process at a time, engine with 1 worker, N = 2,000.

### 9a. Engine on msynd_view after F1c (2026-10-09)
- Full frame: 196 runs; 139 stale, 27 observed, 30 error. The S KeyError is gone.
- All 30 errors are `route times and distances must be finite and nondecreasing`, from 26 trains such as 11007. Passing stations enriched from MMT pages carry sched_off values below the previous halt's scheduled dep (KYN dep 475, next pass KYNX 472).
- **Decision needed:** clamp pass sched_off to the running max during enrichment (store), drop non-monotone passes, or leave these as error runs. Not fixed; it doesn't block the preview.

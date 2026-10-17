# Deployment (monitor at CP-3/G5)

Use the user's Oracle A1 VM with Docker Compose v2+ and the repo at `/opt/msynd`.
Set its timezone to UTC (`sudo timedatectl set-timezone UTC`) for the cron times below.
In both the VM firewall and Oracle ingress rules, open TCP 80/443 (UDP 443 for HTTP/3).
Restrict TCP 22 to the user's IP. Keep PostgreSQL and API ports private.
Point the user's DuckDNS hostname at the VM; maintain its DNS record separately.

Copy `.env.example` to `.env`, replace the `${...}` hostname/auth placeholders, and `chmod 600 .env`.
Generate the password hash with `docker run --rm -it caddy:2 caddy hash-password`.
Use `BASIC_AUTH_HASH='<hash>'` with single quotes in `.env` to preserve bcrypt dollar signs.
For off-site backups, set `OCI_BUCKET_NAME` and place its config/key under `OCI_CONFIG_DIR` (default `/data/oci`).
The OCI config's `key_file` must use the container path `/root/.oci/<key-file>`.
These are user secrets; never commit them. Compose supplies DATABASE_URL for each service's network.

The existing db binding `127.0.0.1:55432:5432` and named `pgdata` volume are unchanged.
Do not move the pilot's database to `/data/pg`; any VM storage migration belongs to the monitor.
Create `/data/{models,backups,tiles,caddy,caddy-config,oci}` and populate model artifacts/tiles locally.
The monitor handles approved imports, migrations and restores; this task performs none.
Models must be registered with paths under `/data/models`; export the network into `data/static`.
Build the dashboard with `npm --prefix web ci` then `npm --prefix web run build`.
Caddy mounts `web/dist` and `/data/tiles` read-only; PMTiles use `/tiles/india.pmtiles`.
All public routes require basic auth, including tiles and WebSocket upgrades.
Caddy obtains TLS certificates automatically and retains them in `/data/caddy`.

## Build and verify (no image push)

`docker buildx build --platform linux/arm64,linux/amd64 .`
`docker compose --profile app --profile ops build`
`docker compose --profile ops run --rm --no-deps ops python -c "import lightgbm, onnxruntime, numba"`
The image uses frozen uv dependencies, ops extras, OpenMP libraries and a PostgreSQL dump client.
Its build context excludes local evidence, backups, secrets and Git metadata.
The monitor must run the pinned-image smoke test on A1 and the measured G5 restore drill.

## Direct cloud collection (2026-10-16)

The collector uses its own `msynd-collector:local` image, built on Python 3.12 Bookworm
(OpenSSL 3.0). The Oracle VM can reach MMT with this client; the Trixie/OpenSSL 3.5
client timed out on the same VM. The laptop relay is no longer required.
The other services keep the Trixie image, including the PostgreSQL 16-compatible dump client.

The normal path is `Oracle collector -> MMT (HTTPS)`. The database, engine, API and
dashboard also run on Oracle, so collection continues when the laptop is shut down.
Use `COMPOSE_FILE=compose.yaml`; the collector must have no `MSYND_RELAY` setting.
The Windows `msynd-relay` scheduled task is disabled in this deployment.

Recorded verification (2026-10-17 13:08 UTC): about 19 hours of independent operation,
3,125 successful requests, zero challenges, and new actual-event revisions stored.
The old VM-IP restriction was an inference from the earlier client failures;
it is superseded by these direct requests from the same VM. No extra cloud account
or paid relay was created. See Task 78 in the implementation plan for the full counts.

For an existing relay deployment, back up `.env`, set `COMPOSE_FILE=compose.yaml`,
and preserve the approved live flag, rate and budgets. Build and switch only the collector:
```sh
docker compose --profile app build collector
docker compose --profile app up -d --no-deps collector
docker update --restart unless-stopped msynd-db-1
```
The last command applies the database restart policy without recreating its container.
Docker must be enabled at boot. Check new `obs.fetch` successes and source freshness with
the laptop relay stopped before disabling its scheduled task. Do not reboot just to test this.
On a new deployment, Compose applies the database restart policy automatically.
Check `source.poll_ok_at` and the MMT dispatcher state in `/api/health`: recent engine
frames alone do not prove that new source data is arriving.

To keep an existing laptop relay task off, run in elevated Windows PowerShell:
```powershell
Disable-ScheduledTask -TaskName msynd-relay
Stop-ScheduledTask -TaskName msynd-relay
Get-ScheduledTask -TaskName msynd-relay | Select-Object TaskName,State
```
Disabling preserves the task for rollback. Stopping it alone permits the next trigger
to restart it. These commands are already applied to the current deployment.

Rollback of the October 16 cutover: in elevated Windows PowerShell run
`Enable-ScheduledTask -TaskName msynd-relay` and `Start-ScheduledTask -TaskName msynd-relay`,
then run on the VM:
```sh
sh /opt/msynd/deploy-backups/20261016-direct-collector/rollback.sh
```
The script restores the saved `.env`, `compose.yaml` and `Dockerfile`, retags the
previous Trixie image as `msynd:local`, and recreates only the collector. Restoring
`.env` alone does not select the old image. Retain the backup and previous image;
review the backup first if configuration has changed since that cutover.
Upstream acceptance can change; investigate fresh failures before assuming an IP restriction.

## Residential relay (fallback)

Use this section only when deliberately reverting to the laptop relay. Normal direct
cloud collection uses the section above and does not require a powered laptop.
The PC's plain httpx client makes each HTTPS request to MMT; the VM dispatcher owns pacing, budgets and retries.
On the PC, open elevated Windows PowerShell in the repo and set `$SshTarget` to your SSH destination:
```powershell
.\relay-task.ps1 -SshTarget $SshTarget
Get-ScheduledTask -TaskName msynd-relay | Get-ScheduledTaskInfo
Get-Content "$env:LOCALAPPDATA\msynd\relay.log" -Tail 30
Stop-ScheduledTask -TaskName msynd-relay
.\relay-task.ps1 -Unregister
```
Registration replaces the task; override `-Identity` or `-Log` if needed. It starts without a login and checks every 5 min.
Stopping lasts until the next trigger; disable it to retain it for rollback, or unregister to remove it.
Relay exit also kills its SSH child on Windows.
Only this fallback requires the PC to stay powered and awake; Windows sleep is inhibited while its relay runs.
On the VM, write `/etc/ssh/sshd_config.d/70-msynd-keepalive.conf` with `ClientAliveInterval 30` and `ClientAliveCountMax 3` on separate lines.
Validate and reload: `sudo sshd -t && sudo systemctl reload ssh`.
For the relay fallback, change only this VM `.env` line, preserving the already-approved
live flag, rate and poll/harvest budgets:
```dotenv
COMPOSE_FILE=compose.yaml:compose.relay.yaml
```
Then use the Startup commands below. When the tunnel is down, requests fail locally as `transport_error`; the dispatcher breaker paces them.

## Startup

Collector, engine, API and Caddy require the `app` profile; ops uses the separate `ops` profile.
`docker compose up -d db` starts only db, preserving the current pilot setup.
Do not start collector while the pilot owns the MMT host lock.
Leave MSYND_LIVE at 0 until the relevant gate; the monitor then uses the direct cloud setup above.
The collector command reads its poll and harvest budgets from `.env` (defaults: 5000 and 0).
The engine main uses run_loop defaults, including its default particle counts and empty S tables.
After the gate and pilot handover, the monitor starts daemons with `docker compose --profile app up -d`.
Review `docker compose --profile app ps` and `docker compose --profile app logs --tail=100`.
Health probes check collector DB connectivity, recent engine frames, API health and Caddy config.
Daemon logs rotate at 10 MB × 5; ops is one-shot and has no db startup dependency.

## Cron (UTC; enable only after monitor deployment)

```cron
0 21 * * * cd /opt/msynd && docker compose --profile ops run --rm --no-deps ops python -m msynd.ops.nightly >> /data/logs/cron-nightly.log 2>&1
30 20 * * 0 cd /opt/msynd && docker compose --profile ops run --rm --no-deps ops python -m msynd.ops.weekly evaluate --models-dir /data/models >> /data/logs/cron-weekly.log 2>&1
```
Cron has no mail transport here, so each job appends to `/data/logs` (create it owned by the cron user).
`/etc/logrotate.d/msynd-cron` rotates them: `/data/logs/cron-*.log { weekly rotate 8 compress missingok notifempty copytruncate }`.
Weekly evaluation runs Monday 02:00 IST (Sunday 20:30 UTC).
An empty `OCI_BUCKET_NAME` means local-only backups with two retained copies.
Local backups retain evidence; when enabled, off-site dumps exclude obs. No raw source export is approved.
Never use `docker compose down -v`: it deletes the database volume.

OSM refresh (monitor; use the verified local India extract):
```sh
docker run --rm -v /data/osm:/data/osm debian:bookworm-slim sh -ec 'apt-get update && apt-get install -y --no-install-recommends osmium-tool && osmium tags-filter /data/osm/src/india-latest.osm.pbf nw/railway=rail,narrow_gauge n/railway=station,halt -o /data/osm/filtered.osm.pbf --overwrite &&
osmium add-locations-to-ways /data/osm/filtered.osm.pbf -o /data/osm/located.osm.pbf --overwrite &&
osmium cat /data/osm/located.osm.pbf -f opl,add_metadata=false,locations_on_ways=true -o /data/osm/rail.opl --overwrite'
docker compose --profile ops run --rm --no-deps -v "$PWD/data/static:/app/data/static" --entrypoint sh ops -ec 'printf "/app\n" > /app/.venv/lib/python3.12/site-packages/msynd.pth; python -I -m msynd.db.migrate
python -I -m msynd.static.osm --opl /data/osm/rail.opl
python -I -m msynd.static.export export'
```
Tiles cleanup (monitor; existing Go `pmtiles` binary; keep `india.pmtiles.orig` for rollback):
```sh
docker compose --profile ops run --rm --no-deps --entrypoint sh ops -ec 'printf "/app\n" > /app/.venv/lib/python3.12/site-packages/msynd.pth; python -I -m msynd.static.tiles /data/tiles/india.pmtiles /data/tiles/india.clean.mbtiles'
pmtiles convert /data/tiles/india.clean.mbtiles /data/tiles/india.clean.pmtiles && pmtiles verify /data/tiles/india.clean.pmtiles
(test -e /data/tiles/india.pmtiles.orig || cp -p /data/tiles/india.pmtiles /data/tiles/india.pmtiles.orig) && mv /data/tiles/india.clean.pmtiles /data/tiles/india.pmtiles
```

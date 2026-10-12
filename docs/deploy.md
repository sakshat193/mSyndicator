# Deployment (monitor at CP-3/G5)

Use the user's Oracle A1 VM with Docker Compose v2+ and the repo at `/opt/msynd`.
Set its timezone to UTC (`sudo timedatectl set-timezone UTC`) for the cron times below.
In both the VM firewall and Oracle ingress rules, open TCP 80/443 (UDP 443 for HTTP/3).
Restrict TCP 22 to the user's IP. Keep PostgreSQL and API ports private.
Point the user's DuckDNS hostname at the VM; maintain its DNS record separately.

Copy `.env.example` to `.env`, replace the `${...}` hostname/auth placeholders, and `chmod 600 .env`.
Generate the password hash with `docker run --rm -it caddy:2 caddy hash-password`.
Use `BASIC_AUTH_HASH='<hash>'` with single quotes in `.env` to preserve bcrypt dollar signs.
Set the OCI bucket and place its config/key under `OCI_CONFIG_DIR` (default `/data/oci`).
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

## Residential relay

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
Stopping lasts until the next trigger; unregister to keep it stopped. Relay exit also kills its SSH child on Windows.
Keep the PC powered with the lid open; Windows sleep is inhibited while the relay runs.
On the VM, write `/etc/ssh/sshd_config.d/70-msynd-keepalive.conf` with `ClientAliveInterval 30` and `ClientAliveCountMax 3` on separate lines.
Validate and reload: `sudo sshd -t && sudo systemctl reload ssh`.
After the gate and pilot handover, set these VM `.env` lines (harvest stays 0 until an approved ramp):
```dotenv
COMPOSE_FILE=compose.yaml:compose.relay.yaml
MSYND_LIVE=1
MMT_HARVEST_BUDGET=0
```
Then use the Startup commands below. When the tunnel is down, requests fail locally as `transport_error`; the dispatcher breaker paces them.

## Startup

Collector, engine, API and Caddy require the `app` profile; ops uses the separate `ops` profile.
`docker compose up -d db` starts only db, preserving the current pilot setup.
Do not start collector while the pilot owns the MMT host lock.
Leave MSYND_LIVE at 0 until the relevant gate; the monitor then uses the relay settings above.
The collector command reads its poll and harvest budgets from `.env` (defaults: 5000 and 0).
The engine main uses run_loop defaults, including its default particle counts and empty S tables.
After the gate and pilot handover, the monitor starts daemons with `docker compose --profile app up -d`.
Review `docker compose --profile app ps` and `docker compose --profile app logs --tail=100`.
Health probes check collector DB connectivity, recent engine frames, API health and Caddy config.
Daemon logs rotate at 10 MB × 5; ops is one-shot and has no db startup dependency.

## Cron (UTC; enable only after monitor deployment)

```cron
0 21 * * * cd /opt/msynd && docker compose --profile ops run --rm --no-deps ops python -m msynd.ops.nightly
30 20 * * 6 cd /opt/msynd && docker compose --profile ops run --rm --no-deps ops python -m msynd.ops.weekly
```
Enable the weekly line only when Task 38 supplies `msynd.ops.weekly`.
Local backups retain evidence; off-site dumps exclude obs. No raw source export is approved.
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

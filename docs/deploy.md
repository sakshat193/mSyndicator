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
These are user secrets; never commit them. Compose overrides DATABASE_URL with the private db hostname.

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

## Startup

Collector, engine, API and Caddy require the `app` profile; ops uses the separate `ops` profile.
`docker compose up -d db` starts only db, preserving the current pilot setup.
Do not start collector while the pilot owns the MMT host lock.
Leave MSYND_LIVE at 0; only the monitor enables collection after the relevant gate.
The collector command is supplied by Task 22; its absence here does not block building the image.
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

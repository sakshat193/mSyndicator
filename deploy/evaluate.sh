#!/bin/sh
# Manual and cron evaluations share this host lock and pinned image.
set -eu
image=$1
job=$2
shift 2
case "$job" in
  weekly) set -- python -m msynd.ops.weekly evaluate --models-dir /data/models ;;
  shadow) set -- python -m msynd.ops.shadow score "$@" ;;
  *) exit 64 ;;
esac
cd /opt/msynd
exec 9>/data/logs/evaluation.lock
flock -n 9 || exit 75
image=$(docker image inspect --format '{{.Id}}' "$image")
name=msynd-evaluation
# Refuse to overwrite an orphan; recover its identity and transaction first.
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "evaluation container already exists" >&2
  exit 75
fi
pw=$(sed -n "s/^POSTGRES_PASSWORD=//p" .env | tr -d "\"'")
test -n "$pw"
export DATABASE_URL="postgresql://msynd:${pw}@db:5432/msynd"
status=0
worker=
cleanup() {
  trap - EXIT HUP INT TERM
  if test -n "$worker"; then kill -TERM "$worker" 2>/dev/null || true; fi
  if docker container inspect "$name" >/dev/null 2>&1; then
    docker stop --timeout 60 "$name" >/dev/null 2>&1 || true
    docker rm -f "$name" >/dev/null 2>&1 || true
  fi
  if test -n "$worker"; then wait "$worker" 2>/dev/null || true; fi
  if docker container inspect "$name" >/dev/null 2>&1; then status=1; fi
  printf '{"event":"finish","time":"%s","job":"%s","image":"%s","exit_status":%s}\n' \
    "$(date -u +%FT%TZ)" "$job" "$image" "$status" >> /data/logs/evaluations.jsonl
  exit "$status"
}
trap cleanup EXIT
trap 'status=143; exit "$status"' HUP INT TERM
printf '{"event":"start","time":"%s","job":"%s","image":"%s"}\n' \
  "$(date -u +%FT%TZ)" "$job" "$image" >> /data/logs/evaluations.jsonl
timeout --signal=TERM --kill-after=60s 24h \
  docker run --rm --init --name "$name" --stop-timeout 60 --network msynd_default \
    -v /data:/data -e DATABASE_URL -e MSYND_LIVE=0 "$image" "$@" 9>&- &
worker=$!
wait "$worker" || status=$?
worker=
exit "$status"

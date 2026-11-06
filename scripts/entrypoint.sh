#!/usr/bin/env bash
# Container entrypoint (runs under tini, so SIGTERM/SIGINT reach Python and zombies are reaped).
# Starts as root ONLY to make the cache dirs writable, then re-execs itself as the unprivileged 'mme' user.
set -euo pipefail

export PYTHONUNBUFFERED=1
: "${HF_HOME:=/workspace/.cache/huggingface}"
: "${TRITON_CACHE_DIR:=${HF_HOME%/huggingface}/triton}"
export HF_HOME TRITON_CACHE_DIR

APP_UID="${MME_UID:-10001}"

if [ "$(id -u)" = "0" ] && [ "${MME_RUN_AS_ROOT:-0}" != "1" ]; then
  mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"
  # RunPod volumes are usually root-owned. Fix ownership only when something is not owned by the app user
  # (avoids a recursive chown of a multi-GB cache on every start).
  for d in "$HF_HOME" "$TRITON_CACHE_DIR"; do
    if [ -n "$(find "$d" -maxdepth 4 ! -uid "$APP_UID" -print -quit 2>/dev/null)" ]; then
      echo "[entrypoint] chown -R $APP_UID $d" >&2
      chown -R "$APP_UID:$APP_UID" "$d" || echo "[entrypoint] WARNING: chown failed for $d" >&2
    fi
  done
  export HOME=/home/mme
  # exec keeps the PID, so signals from tini still reach the final process
  exec setpriv --reuid="$APP_UID" --regid="$APP_UID" --clear-groups "$0" "$@"
fi

cd /app
if [ "${MME_SKIP_STARTUP_CHECK:-0}" != "1" ]; then
  python /app/scripts/startup_check.py      # fails closed: bad config / no GPU / unwritable cache -> exit 1
fi

# The server opens the port immediately; /ping is 204 while the model downloads/loads/warms, 200 when ready.
exec python -m server.api "$@"

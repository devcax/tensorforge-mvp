#!/usr/bin/env bash
set +x
set -euo pipefail

stage="initialization"
trap 'printf "::error::Deployment failed during %s.\n" "$stage" >&2' ERR

ssh_run() {
  local remote_command
  remote_command="$(python3 -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@")"
  ssh -p "$PORT" -i ~/.ssh/id_ed25519 \
    -o BatchMode=yes -o StrictHostKeyChecking=yes -o LogLevel=ERROR \
    -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=20 \
    "$SSH_USER@$SSH_HOST" "$remote_command"
}

case "${1:-}" in
  check-settings)
    stage="settings validation"
    for name in TS_AUTHKEY SSH_PRIVATE_KEY SSH_USER SSH_HOST WORK_DIR PORT ENV_VAR; do
      if [ -z "${!name:-}" ]; then
        printf '::error::Missing secret %s in environment ENV or repository settings.\n' "$name" >&2
        exit 1
      fi
    done
    if [[ ! "$PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$PORT < 1 || 10#$PORT > 65535 )); then
      echo "::error::PORT must be an SSH port between 1 and 65535." >&2
      exit 1
    fi
    if [[ "$WORK_DIR" != /* || "$WORK_DIR" == / ]]; then
      echo "::error::WORK_DIR must be an absolute application directory." >&2
      exit 1
    fi
    ;;
  setup-ssh)
    stage="SSH setup"
    umask 077
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    printf '%s\n' "$SSH_PRIVATE_KEY" > ~/.ssh/id_ed25519
    chmod 600 ~/.ssh/id_ed25519
    ssh-keyscan -T 20 -p "$PORT" -H "$SSH_HOST" > ~/.ssh/known_hosts 2>/dev/null
    test -s ~/.ssh/known_hosts
    chmod 600 ~/.ssh/known_hosts
    ;;
  deploy)
    stage="server prerequisites"
    ssh_run bash -s -- "$WORK_DIR" >/dev/null 2>&1 <<'EOF'
set -euo pipefail
case "$(uname -m)" in
  aarch64|arm64) ;;
  *) exit 1 ;;
esac
docker info >/dev/null
docker compose version >/dev/null
mkdir -p -- "$1"
EOF

    stage="image transfer"
    gzip -c "$RUNNER_TEMP/tensorforge-api.tar" | \
      ssh_run bash -c 'set -euo pipefail; gunzip -c | docker load --quiet' >/dev/null 2>&1

    stage="Compose configuration"
    ssh_run bash -c 'set -euo pipefail; cat > "$1/docker-compose.yml.new"; mv -- "$1/docker-compose.yml.new" "$1/docker-compose.yml"' \
      -- "$WORK_DIR" < docker-compose.yml >/dev/null 2>&1

    stage="application environment"
    printf '%s\n' "$ENV_VAR" | \
      ssh_run bash -c 'set -euo pipefail; umask 077; cat > "$1/.env.new"; chmod 600 "$1/.env.new"; mv -- "$1/.env.new" "$1/.env"' \
        -- "$WORK_DIR" >/dev/null 2>&1

    stage="container startup and readiness"
    echo "Waiting for application readiness..."
    ssh_run bash -s -- "$WORK_DIR" >/dev/null 2>&1 <<'EOF'
set -euo pipefail
cd -- "$1"
docker compose config --quiet
docker compose up -d --no-build --pull never --remove-orphans --wait --wait-timeout 1800 api
EOF
    echo "Deployment finished successfully."
    ;;
  cleanup)
    stage="SSH cleanup"
    rm -f ~/.ssh/id_ed25519
    ;;
  *)
    echo "::error::Unknown deployment operation." >&2
    exit 1
    ;;
esac

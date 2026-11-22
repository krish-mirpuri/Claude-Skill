#!/usr/bin/env bash
# Reproducible self-hosted SupoClip install, with a local keyless LLM.
#
# Verified on a Claude Code cloud session (Ubuntu-ish container, 15 GB RAM,
# no GPU). On a normal laptop/VPS most of the cloud-specific bits are skipped
# automatically. Run from anywhere:   bash install.sh
#
# What it sets up:
#   - SupoClip from source via docker compose (frontend/backend/worker/mcp/pg/redis)
#   - local Whisper for transcription   (no API key)
#   - local Ollama + qwen2.5:7b-instruct for clip selection (no API key)
#   - a verified local account + an API key, so scripts can drive it
set -euo pipefail

ROOT="${SUPOCLIP_ROOT:-$HOME/supoclip}"
OLLAMA_DIR="${OLLAMA_DIR:-$HOME/ollama}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$HOME/ollama-models}"
MODEL_BASE="${MODEL_BASE:-qwen2.5:7b-instruct}"
MODEL_NAME="${MODEL_NAME:-clipper}"
NUM_CTX="${NUM_CTX:-16384}"
ACCOUNT_EMAIL="${ACCOUNT_EMAIL:-owner@supoclip.local}"
CA_BUNDLE="/root/.ccr/ca-bundle.crt"   # only exists inside a Claude cloud session

say() { printf '\n==> %s\n' "$*"; }

say "1/8  checking prerequisites"
command -v git >/dev/null || { echo "git is required"; exit 1; }
command -v docker >/dev/null || { echo "docker is required"; exit 1; }
if ! docker info >/dev/null 2>&1; then
  say "docker daemon is not running; starting dockerd"
  sudo -n dockerd >/tmp/dockerd.log 2>&1 &
  for _ in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done
  docker info >/dev/null 2>&1 || { echo "could not start the docker daemon; see /tmp/dockerd.log"; exit 1; }
fi

say "2/8  cloning SupoClip into $ROOT"
if [ -d "$ROOT/.git" ]; then
  echo "already present, leaving it alone"
else
  git clone --depth 1 https://github.com/FujiwaraChoki/supoclip.git "$ROOT"
fi
cd "$ROOT"

# --- cloud-session only: containers must trust the egress proxy's CA ----------
if [ -f "$CA_BUNDLE" ]; then
  say "3/8  injecting the egress-proxy CA into the images (cloud session detected)"
  python3 - "$CA_BUNDLE" "$ROOT" <<'PY'
import pathlib, re, shutil, sys
ca_src, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
CA_NAME, CA_DST = "ccr-ca-bundle.crt", "/usr/local/share/ca-certificates/ccr-ca-bundle.crt"
BLOCK = f"""
# >>> proxy-ca >>>
COPY {CA_NAME} {CA_DST}
RUN cat {CA_DST} >> /etc/ssl/certs/ca-certificates.crt || true
ENV SSL_CERT_FILE={CA_DST} \\
    REQUESTS_CA_BUNDLE={CA_DST} \\
    CURL_CA_BUNDLE={CA_DST} \\
    PIP_CERT={CA_DST} \\
    NODE_EXTRA_CA_CERTS={CA_DST} \\
    DENO_CERT={CA_DST} \\
    GIT_SSL_CAINFO={CA_DST} \\
    UV_NATIVE_TLS=1
# <<< proxy-ca <<<
"""
for df, ctx in [("frontend/Dockerfile", "."), ("backend/Dockerfile", "backend"), ("mcp/Dockerfile", "mcp")]:
    p = root / df
    t = p.read_text()
    if "proxy-ca" in t:
        continue
    out = []
    for line in t.splitlines(keepends=True):
        out.append(line)
        m = re.match(r"^FROM\s+(\S+)", line)
        if m and (":" in m.group(1) or "/" in m.group(1)):
            out.append(BLOCK)
    t = "".join(out)
    # yt-dlp / requests read certifi's own bundle, not the system store
    anchor = "# yt-dlp and its PO-token plugin are installed from uv.lock above."
    if df == "backend/Dockerfile" and anchor in t:
        t = t.replace(anchor, (
            'RUN for f in $(find /app/.venv -name cacert.pem); do '
            'cat %s >> "$f"; done\n\n' % CA_DST) + anchor, 1)
    p.write_text(t)
    shutil.copy2(ca_src, root / ctx / CA_NAME)
    print("patched", df)
PY
  # let backend/worker reach an Ollama server on the host
  cat > docker-compose.override.yml <<'YML'
services:
  backend:
    extra_hosts: ["host.docker.internal:host-gateway"]
  worker:
    extra_hosts: ["host.docker.internal:host-gateway"]
  ai-worker:
    extra_hosts: ["host.docker.internal:host-gateway"]
YML
else
  say "3/8  no egress proxy detected, skipping CA injection"
  cat > docker-compose.override.yml <<'YML'
services:
  backend:
    extra_hosts: ["host.docker.internal:host-gateway"]
  worker:
    extra_hosts: ["host.docker.internal:host-gateway"]
  ai-worker:
    extra_hosts: ["host.docker.internal:host-gateway"]
YML
fi

say "4/8  writing .env"
if [ -f .env ]; then
  echo ".env already exists, leaving it alone"
else
  cp .env.example .env
  python3 - <<PY
import re, pathlib, subprocess
def secret():
    return subprocess.check_output(["openssl","rand","-base64","32"]).decode().strip()
p = pathlib.Path(".env"); t = p.read_text()
vals = {
  "TRANSCRIPTION_PROVIDER": "whisper",
  "WHISPER_MODEL_SIZE": "medium",
  "TRANSCRIPTION_LANGUAGE": "en",
  "BACKEND_CPU_ONLY": "true",
  "LLM": "ollama:${MODEL_NAME}",
  "OLLAMA_BASE_URL": "http://host.docker.internal:11434/v1",
  "YOUTUBE_POT_PROVIDER_URL": "http://bgutil:4416",
  "FRONTEND_PORT": "3107",
  "DISABLE_SIGN_UP": "false",
  "BETTER_AUTH_SECRET": secret(),
  "BACKEND_AUTH_SECRET": secret(),
  "APP_SETTINGS_ENCRYPTION_KEY": secret(),
}
for k, v in vals.items():
    pat = re.compile(rf"^{re.escape(k)}=.*$", re.M)
    t = pat.sub(f"{k}={v}", t, count=1) if pat.search(t) else t + f"\n{k}={v}\n"
p.write_text(t)
print("wrote .env")
PY
fi

say "5/8  installing Ollama into $OLLAMA_DIR (no API key needed)"
if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
  command -v zstd >/dev/null || sudo -n apt-get install -y zstd || true
  mkdir -p "$OLLAMA_DIR"
  curl -fL --retry 3 https://ollama.com/download/ollama-linux-amd64.tar.zst \
    | zstd -d | tar -xf - -C "$OLLAMA_DIR"
fi
mkdir -p "$OLLAMA_MODELS_DIR"
if ! curl -sf --noproxy '*' --max-time 5 http://localhost:11434/api/version >/dev/null; then
  OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS="$OLLAMA_MODELS_DIR" \
  OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=10m \
  LD_LIBRARY_PATH="$OLLAMA_DIR/lib" \
    setsid nohup "$OLLAMA_DIR/bin/ollama" serve > "$HOME/ollama-serve.log" 2>&1 < /dev/null &
  for _ in $(seq 1 30); do
    curl -sf --noproxy '*' --max-time 5 http://localhost:11434/api/version >/dev/null && break
    sleep 2
  done
fi
export OLLAMA_HOST=127.0.0.1:11434
"$OLLAMA_DIR/bin/ollama" pull "$MODEL_BASE"
printf 'FROM %s\nPARAMETER num_ctx %s\n' "$MODEL_BASE" "$NUM_CTX" > /tmp/Modelfile.clipper
"$OLLAMA_DIR/bin/ollama" create "$MODEL_NAME" -f /tmp/Modelfile.clipper

say "6/8  building images (first run takes 10-20 minutes)"
docker compose build

say "7/8  starting the stack"
COMPOSE_PROFILES=youtube-proxy docker compose up -d
for _ in $(seq 1 60); do
  curl -sf --noproxy '*' --max-time 5 http://localhost:8000/health/db >/dev/null && break
  sleep 5
done
curl -sS --noproxy '*' http://localhost:8000/health/db; echo

say "8/8  creating a verified account and an API key"
PWFILE="$HOME/supoclip-account-password.txt"
KEYFILE="$HOME/supoclip-api-key.txt"
if [ ! -f "$KEYFILE" ]; then
  umask 077
  openssl rand -base64 18 | tr -d '/+=' | head -c 20 > "$PWFILE"
  PW=$(cat "$PWFILE")
  curl -sS --noproxy '*' --max-time 60 -X POST http://localhost:3107/api/auth/sign-up/email \
    -H 'Content-Type: application/json' \
    -d "{\"email\":\"$ACCOUNT_EMAIL\",\"password\":\"$PW\",\"name\":\"Owner\"}" >/dev/null
  # self-hosted installs have no SES, so verify the address directly
  docker compose exec -T postgres psql -U supoclip -d supoclip \
    -c "update users set \"emailVerified\"=true where email='$ACCOUNT_EMAIL';" >/dev/null
  J=$(mktemp)
  curl -sS --noproxy '*' --max-time 60 -c "$J" -X POST http://localhost:3107/api/auth/sign-in/email \
    -H 'Content-Type: application/json' \
    -d "{\"email\":\"$ACCOUNT_EMAIL\",\"password\":\"$PW\"}" >/dev/null
  curl -sS --noproxy '*' --max-time 60 -b "$J" -X POST http://localhost:3107/api/api-keys \
    -H 'Content-Type: application/json' -d '{"name":"automation"}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["api_key"]["key"])' > "$KEYFILE"
  rm -f "$J"
fi
echo "API key saved to $KEYFILE"

cat <<DONE

SupoClip is up.
  UI        http://localhost:3107     (sign in as $ACCOUNT_EMAIL, password in $PWFILE)
  API docs  http://localhost:8000/docs
  make clips:  python3 make_clips.py "<youtube-url-or-local-file>"

Before exposing this to anyone else, change the Postgres password: it is
hardcoded as supoclip_password in four places in docker-compose.yml.
DONE

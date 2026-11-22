---
name: clip-video
description: Turn a long video into short vertical clips with captions, using the self-hosted SupoClip stack in this repo. Use this whenever the user hands over a video and wants shorts, clips, highlights, reels or TikToks out of it — including when they just paste a YouTube/Twitch/Kick link or attach a video file with something like "clip this", "make shorts from this", "find the best bits" or "what's worth posting here". Also use it when they ask to install, start, restart or check on SupoClip. The user should never have to run the commands themselves; running them is this skill's job.
---

# Clip a video into shorts

The point of this skill is that the user sends a video and gets clips back. They
should not be asked to run commands, read logs, or debug Docker. When something
breaks, fix it silently if you can, and only surface what genuinely needs a
human decision.

## What the stack is

SupoClip, self-hosted, in `~/supoclip`, driven by two files in this repo:

- `supoclip/install.sh` — installs everything from scratch and is safe to re-run.
- `supoclip/make_clips.py` — takes a video, returns rendered clips.

It is deliberately keyless: local Whisper transcribes, a local Ollama model
(`clipper`) picks the moments. Nothing is sent to a third party.

## Start here every time

Run the preflight before anything else, because this stack has three moving
parts that each die independently — and an idle cloud container reclaims all of
them, so "it worked an hour ago" means nothing.

```bash
docker info >/dev/null 2>&1 && echo "docker ok" || echo "docker DOWN"
curl -sf --noproxy '*' --max-time 5 http://localhost:11434/api/version >/dev/null && echo "ollama ok" || echo "ollama DOWN"
curl -sf --noproxy '*' --max-time 5 http://localhost:8000/health/db && echo || echo "supoclip DOWN"
```

Repair whatever is down, in this order:

**Docker daemon down** — common in cloud containers, where nothing starts it:
```bash
sudo -n dockerd >/tmp/dockerd.log 2>&1 &
```
Give it ~20s, then re-check `docker info`.

**Ollama down** — it must bind `0.0.0.0` so the containers can reach it, and it
must stay capped at one model. Two 8B models resident at once will exhaust a
15 GB box and the kernel starts killing things:
```bash
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=$HOME/ollama-models \
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=10m \
LD_LIBRARY_PATH=$HOME/ollama/lib \
  setsid nohup $HOME/ollama/bin/ollama serve > $HOME/ollama-serve.log 2>&1 < /dev/null &
```

**SupoClip down but `~/supoclip` exists** — just bring it back up:
```bash
cd ~/supoclip && COMPOSE_PROFILES=youtube-proxy docker compose up -d
```

**`~/supoclip` missing entirely** — full install. Tell the user it's a one-time
20–30 minute build so the silence is expected, then run it in the background:
```bash
bash supoclip/install.sh
```

## Making the clips

```bash
python3 supoclip/make_clips.py "<url-or-file>" --clip-length short
```

`--clip-length` is `auto` (15–60s), `short` (15–30s), `medium` (30–60s) or
`long` (60–120s). Pass `--instructions "..."` when the user says what they're
looking for ("the funny bits", "anything about pricing") — it steers selection
and costs nothing.

Run it in the background and report progress as it moves through its stages,
because a feature-length source can take a long time and silence reads as a
hang. Transcription is the slow part on CPU: budget roughly two minutes per
minute of video.

When it finishes, send the user the actual clip files rather than describing
them or printing paths. Read the `clips.json` written alongside them and lead
with what the AI thought — the title, and the hook/engagement/value scores.

**Check `hook_type` before you celebrate.** If it reads `fallback` with
near-zero scores, the AI found nothing worth clipping and SupoClip just cut the
opening of the video. That is a failure wearing a success's clothes. The usual
cause is a bad transcript, so look at `clips.json`'s `text` field: if it's
gibberish or drifting between alphabets, fix the transcription rather than
re-rolling the clip. Set `TRANSCRIPTION_LANGUAGE` in `~/supoclip/.env` to the
actual spoken language and raise `WHISPER_MODEL_SIZE` to `medium`, then restart
`backend worker ai-worker` and try again.

## YouTube links from a datacenter IP

YouTube serves metadata and solves the JS challenge happily, then refuses the
media itself with `403 Forbidden` when the request comes from a cloud host. The
PO-token sidecar is already enabled and does not help; SupoClip's own docs say
PO tokens "do not bypass an IP block on their own."

Do not try to work around this by spoofing player clients or swapping formats —
that is circumventing an access control, not a configuration fix.

Instead, confirm it quickly rather than guessing, since the block is per-IP and
a home machine is unaffected:

```bash
cd ~/supoclip && docker compose exec -T worker sh -c 'cd /tmp && /app/.venv/bin/python -m yt_dlp --no-playlist --no-warnings -f "bv*[height<=480]+ba/b[height<=480]" --download-sections "*60-80" -o "probe.%(ext)s" "<url>" 2>&1 | tail -3'
```

If it 403s, say so plainly and offer the two things that do work: the user
attaches the video file instead, or the stack runs on their own machine, where
the same link downloads fine. Don't make them guess which.

## Long videos

Two separate ceilings, and it's worth naming the right one:

- The local model holds 16k tokens, roughly an hour of speech. A longer
  transcript will not fit.
- CPU transcription time grows with length, so a multi-hour source is measured
  in hours.

So for anything beyond about an hour, process a range rather than the whole
thing. The API takes `start_seconds` and `end_seconds` on the source, and
`make_clips.py` can be pointed at a specific window. Ask the user which part
matters, or suggest a window if they don't know — guessing wastes hours.

If they genuinely need the whole of a long video, the honest answer is a hosted
model: set `LLM=google-gla:gemini-3-flash-preview` and `GOOGLE_API_KEY` in
`.env`. Offer it as a tradeoff, not a requirement — the keyless setup is a
deliberate choice, not an oversight.

## Credentials

The API key lives in `~/supoclip-api-key.txt` and `make_clips.py` reads it
itself. Never print it, and never paste it into a commit or a chat message.

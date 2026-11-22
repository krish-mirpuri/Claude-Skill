# SupoClip, self-hosted + automated

[SupoClip](https://github.com/FujiwaraChoki/supoclip) is an open-source AI video
clipper: give it a long video, it transcribes it, picks the standout moments with
an LLM, and renders 9:16 vertical clips with word-synced captions.

This directory holds what is needed to stand it up and drive it from the command
line — no web UI, no API keys.

## Files

| File | What it does |
| --- | --- |
| `install.sh` | One-shot install: clone, configure, build, start, create a verified account and an API key. |
| `make_clips.py` | `python3 make_clips.py <video>` → finished clips on disk. |

## Install

```bash
bash install.sh
```

Takes roughly 20–30 minutes the first time, almost all of it building images and
pulling the LLM. Afterwards:

- UI: `http://localhost:3107`
- API docs: `http://localhost:8000/docs`
- API key: `~/supoclip-api-key.txt`

## Make clips

```bash
python3 make_clips.py /path/to/video.mp4
```

```bash
python3 make_clips.py /path/to/video.mp4 --clip-length short --out ~/clips
```

It uploads the file (or takes a URL), creates the task, prints progress as the
pipeline moves through download → transcription → selection → render, then
downloads every finished clip plus a `clips.json` of their titles and scores.

`--clip-length` is `auto` (15–60s), `short` (15–30s), `medium` (30–60s) or
`long` (60–120s). `--instructions "..."` steers what the AI looks for.

## Driving it from Claude Code

SupoClip ships an MCP server, running here on `127.0.0.1:9100`. Pointing Claude
Code at it means you can just say "clip this video" instead of running the
script yourself:

```bash
claude mcp add --transport http supoclip http://localhost:9100/mcp --header "Authorization: Bearer $(cat ~/supoclip-api-key.txt)"
```

The server only answers with a valid key, so an unauthenticated request to
`/mcp` returns 401 — that is the healthy response, not a fault.

## Choices this setup makes

**No API keys.** Transcription uses local Whisper (`TRANSCRIPTION_PROVIDER=whisper`,
`small` model). Clip selection uses a local Ollama model — `qwen2.5:7b-instruct`
rebuilt as `clipper` with a 16k context window, which fits roughly an hour of
speech. SupoClip supports this natively through an OpenAI-compatible endpoint.

A hosted API key is still the better option for quality and speed. To switch,
set in `.env` and restart `backend worker ai-worker`:

```bash
LLM=google-gla:gemini-3-flash-preview
GOOGLE_API_KEY=...
```

**Transcription is pinned to English with the `medium` Whisper model.**
`TRANSCRIPTION_LANGUAGE` defaults to auto-detect, which on noisy or archival
audio drifts mid-video and emits stretches of the wrong alphabet entirely. The
LLM then has nothing usable to select from and SupoClip falls back to clipping
the opening of the video. Pin the language whenever you know it; drop to `small`
only if the box is short on memory.

**One model in memory at a time.** Ollama runs with `OLLAMA_MAX_LOADED_MODELS=1`
and `OLLAMA_NUM_PARALLEL=1`. Two 8B models resident at once exhausted a 15 GB
box and the kernel started killing processes.

**Email verification is bypassed locally.** Self-hosted SupoClip needs SES
configured to deliver verification links, and refuses to mint an API key for an
unverified address. `install.sh` sets `emailVerified` directly in Postgres.

## Known limits

**YouTube and other URL sources may not work from a datacenter IP.** YouTube
serves the metadata and solves the JS challenge fine, then refuses the actual
media with `403 Forbidden`. SupoClip ships a PO-token sidecar (`bgutil`, enabled
here) but its own docs are explicit that PO tokens "do not bypass an IP block on
their own." From a home or office network this is usually not an issue. Where it
is, upload the file instead — that path has no such restriction.

**CPU only.** `BACKEND_CPU_ONLY=true`, no GPU. Transcription and rendering are
several times slower than on a GPU box.

**Postgres password is the upstream default.** `supoclip_password`, hardcoded in
four places in `docker-compose.yml`. Every port binds to `127.0.0.1`, so nothing
is exposed, but change it before putting this anywhere shared.

## Licence

SupoClip is under the **SupoClip Noncommercial Source License** — not an OSI
open-source licence. Noncommercial self-hosting and individual creators
monetising their own channels are fine. Company, agency, client or team use
needs written permission from Shiori LLC (support@supoclip.com).

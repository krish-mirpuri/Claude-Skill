#!/usr/bin/env python3
"""Give it a video (URL or local file); it returns rendered vertical clips.

Drives a local self-hosted SupoClip stack over its REST API:
  upload (if local) -> create task -> poll -> download rendered clips.
"""
import argparse, json, mimetypes, os, sys, time, urllib.error, urllib.request, uuid
from pathlib import Path

BACKEND = os.environ.get("SUPOCLIP_BACKEND", "http://localhost:8000")
KEY_FILE = Path("/home/user/supoclip-api-key.txt")
TERMINAL_OK = {"completed", "complete", "succeeded", "done", "finished"}
TERMINAL_BAD = {"failed", "error", "errored", "cancelled", "canceled"}


def api_key() -> str:
    k = os.environ.get("SUPOCLIP_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not k:
        sys.exit("No API key. Set SUPOCLIP_API_KEY or populate %s" % KEY_FILE)
    return k


def req(method, path, body=None, raw=None, ctype=None, timeout=120):
    url = path if path.startswith("http") else BACKEND.rstrip("/") + path
    data = None
    headers = {"Authorization": "Bearer " + api_key(), "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        data = raw
        headers["Content-Type"] = ctype or "application/octet-stream"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    # never send this through the egress proxy; it is a localhost service
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(r, timeout=timeout) as resp:
        payload = resp.read()
        if resp.headers.get("Content-Type", "").startswith("application/json"):
            return json.loads(payload or b"{}")
        return payload


def upload_local(path: Path) -> str:
    """multipart/form-data upload of a local video; returns the upload:// source url."""
    boundary = "----supoclip" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
    pre = ("--%s\r\nContent-Disposition: form-data; name=\"video\"; filename=\"%s\"\r\n"
           "Content-Type: %s\r\n\r\n" % (boundary, path.name, ctype)).encode()
    post = ("\r\n--%s--\r\n" % boundary).encode()
    body = pre + path.read_bytes() + post
    print("uploading %s (%.1f MB)..." % (path.name, len(body) / 1e6), flush=True)
    out = req("POST", "/upload", raw=body, ctype="multipart/form-data; boundary=" + boundary, timeout=3600)
    for k in ("video_path", "path", "url", "source"):
        if isinstance(out, dict) and out.get(k):
            return out[k]
    sys.exit("upload succeeded but no path in response: %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="YouTube/Twitch/Kick URL, or a local video file path")
    ap.add_argument("--out", default="/home/user/clips", help="where to write finished clips")
    ap.add_argument("--clip-length", default="auto", choices=["auto", "short", "medium", "long"])
    ap.add_argument("--instructions", default=None, help="optional steer for clip selection")
    ap.add_argument("--timeout", type=int, default=7200, help="max seconds to wait")
    a = ap.parse_args()

    p = Path(a.source)
    source_url = upload_local(p) if p.exists() and p.is_file() else a.source

    body = {"source": {"url": source_url}, "clip_length": a.clip_length}
    if a.instructions:
        body["instructions"] = a.instructions
    # YouTube's bot check is intermittent from datacenter IPs; the backend answers
    # 503 VIDEO_METADATA_UNAVAILABLE when it cannot read the duration. Retry a few times.
    task = None
    for attempt in range(1, 7):
        try:
            task = req("POST", "/tasks/", body=body)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code in (503, 429) and attempt < 6:
                print("  create attempt %d got %s; retrying in 15s" % (attempt, e.code), flush=True)
                time.sleep(15)
                continue
            sys.exit("task creation failed (HTTP %s): %s" % (e.code, detail))
    tid = task.get("task_id") or task.get("id")
    if not tid:
        sys.exit("no task id returned: %s" % task)
    print("task %s created" % tid, flush=True)

    start, last = time.time(), None
    while True:
        if time.time() - start > a.timeout:
            sys.exit("timed out after %ss; task %s still running" % (a.timeout, tid))
        try:
            d = req("GET", "/tasks/%s" % tid, timeout=60)
        except urllib.error.HTTPError as e:
            print("poll error %s; retrying" % e.code, flush=True)
            time.sleep(10)
            continue
        status = str(d.get("status") or d.get("state") or "").lower()
        step = d.get("current_step") or d.get("progress_message") or ""
        pct = d.get("progress") or d.get("progress_percent") or ""
        line = "%s %s %s" % (status, pct, step)
        if line != last:
            print("  [%4ds] %s" % (time.time() - start, line.strip()), flush=True)
            last = line
        if status in TERMINAL_OK:
            break
        if status in TERMINAL_BAD:
            sys.exit("task %s ended as %s: %s" % (tid, status, d.get("error") or d.get("error_message") or ""))
        time.sleep(8)

    clips = req("GET", "/tasks/%s/clips" % tid, timeout=120)
    items = clips.get("clips", clips) if isinstance(clips, dict) else clips
    if not items:
        sys.exit("task completed but produced no clips")
    outdir = Path(a.out) / tid
    outdir.mkdir(parents=True, exist_ok=True)
    print("downloading %d clip(s) -> %s" % (len(items), outdir), flush=True)
    written = []
    for i, c in enumerate(items, 1):
        cid = c.get("id") or c.get("clip_id")
        name = c.get("filename") or ("clip_%02d.mp4" % i)
        blob = req("GET", "/tasks/%s/clips/%s/file" % (tid, cid), timeout=1800)
        dest = outdir / name
        dest.write_bytes(blob if isinstance(blob, bytes) else json.dumps(blob).encode())
        written.append(dest)
        title = c.get("title") or c.get("hook") or ""
        print("  %s  (%.1f MB)  %s" % (dest.name, dest.stat().st_size / 1e6, title), flush=True)
    (outdir / "clips.json").write_text(json.dumps(items, indent=2, default=str))
    print("\nDONE: %d clips in %s" % (len(written), outdir))


if __name__ == "__main__":
    main()

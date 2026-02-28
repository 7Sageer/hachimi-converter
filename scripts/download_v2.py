#!/usr/bin/env python3
"""Download paired audio: hachimi from API, original from YouTube search."""

import json
import subprocess
import urllib.request
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PAIRS_FILE = Path(__file__).parent.parent / "pairs_full.jsonl"
PROXY = "http://localhost:10808"


def sanitize_filename(name):
    """Remove characters that are problematic in filenames."""
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '_')
    return name[:80]


def download_hachimi(url, output_path):
    """Download hachimi audio directly from API URL."""
    if output_path.exists():
        return True
    try:
        proxy = urllib.request.ProxyHandler({"https": PROXY, "http": PROXY})
        opener = urllib.request.build_opener(proxy)
        req = urllib.request.Request(url, headers={
            "Referer": "https://hachimi.world/",
            "User-Agent": "HachimiWorld/1.0",
        })
        with opener.open(req, timeout=30) as resp:
            output_path.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"    [fail] hachimi: {e}")
        return False


def download_original(query, output_path):
    """Download original song from YouTube search."""
    if output_path.exists():
        return True
    cmd = [
        "yt-dlp",
        "--proxy", PROXY,
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--max-downloads", "1",
        "-o", str(output_path),
        f"ytsearch1:{query}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return True
    except Exception as e:
        print(f"    [fail] original: {str(e)[:100]}")
        return False


def main():
    (DATA_DIR / "hachimi").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "original").mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in PAIRS_FILE.read_text().strip().split("\n")]
    print(f"Total pairs: {len(pairs)}")

    ok_count = 0
    for i, p in enumerate(pairs):
        name = sanitize_filename(p["name"])
        print(f"\n[{i+1}/{len(pairs)}] {p['name']}")

        # Download hachimi version
        hach_path = DATA_DIR / "hachimi" / f"{name}.mp3"
        h_ok = download_hachimi(p["hachimi_audio_url"], hach_path)
        if h_ok:
            print(f"    [ok] hachimi")

        # Download original
        orig_path = DATA_DIR / "original" / f"{name}.wav"
        query = p.get("youtube_query") or p.get("original_query") or p["original"]
        o_ok = download_original(query, orig_path)
        if o_ok:
            print(f"    [ok] original")

        if h_ok and o_ok:
            ok_count += 1

        time.sleep(1)  # rate limit

    print(f"\nDone. {ok_count}/{len(pairs)} pairs downloaded successfully.")


if __name__ == "__main__":
    main()

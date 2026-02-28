#!/usr/bin/env python3
"""Download original songs using URLs from pairs_with_urls.jsonl."""

import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "original_v2"
PAIRS_FILE = Path(__file__).parent.parent / "pairs_with_urls.jsonl"
PROXY = "http://localhost:10808"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def sanitize(name):
    for ch in '/\\:*?"<>|':
        name = name.replace(ch, '_')
    return name[:80]


def classify_url(url):
    """Return platform type for yt-dlp routing."""
    if not url:
        return None
    url = url.strip()
    # Normalize bare BV ids
    if url.startswith("BV") and " " not in url:
        return "bilibili"
    if "bilibili.com" in url or "b23.tv" in url:
        return "bilibili"
    if "youtube.com" in url or "youtu.be" in url or "music.youtube.com" in url:
        return "youtube"
    if "nicovideo.jp" in url:
        return "nicovideo"
    if "163.com" in url or "163cn.tv" in url:
        return "netease"
    if "y.qq.com" in url or "kugou.com" in url or "kuwo.cn" in url:
        return "china_music"
    if "music.apple.com" in url:
        return "apple"
    return "other"


def normalize_url(url, platform):
    url = url.strip()
    if platform == "bilibili":
        # Handle bare BV ids like "BV1ZG1yYkENc" or "b站：BV1ZG1yYkENc"
        if "BV" in url and "bilibili.com" not in url and "b23.tv" not in url:
            bv = url.split("BV")[-1].split()[0].strip("：: ")
            return f"https://www.bilibili.com/video/BV{bv}"
    return url


def download(url, out_path, platform):
    if out_path.exists():
        return True, "skip"

    url = normalize_url(url, platform)

    # yt-dlp handles bilibili, youtube, nicovideo, netease, etc.
    cmd = [
        "yt-dlp",
        "--proxy", PROXY,
        "-x", "--audio-format", "wav", "--audio-quality", "0",
        "--max-downloads", "1",
        "-o", str(out_path),
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return True, "ok"
    except Exception as e:
        return False, str(e)[:80]


def main():
    pairs = [json.loads(l) for l in PAIRS_FILE.read_text().strip().split("\n")]
    ok, skip, fail, no_url = 0, 0, 0, 0

    for i, p in enumerate(pairs):
        name = sanitize(p["name"])
        out = DATA_DIR / f"{name}.wav"
        url = p.get("orig_url")

        if not url:
            no_url += 1
            print(f"[{i+1}/{len(pairs)}] ⚠️  no url: {p['name'][:40]}")
            continue

        platform = classify_url(url)
        success, note = download(url, out, platform)

        if note == "skip":
            skip += 1
        elif success:
            ok += 1
            print(f"[{i+1}/{len(pairs)}] ✅ {p['name'][:40]}")
        else:
            fail += 1
            print(f"[{i+1}/{len(pairs)}] ❌ {p['name'][:40]}: {note}")

        time.sleep(0.5)

    print(f"\nDone: {ok} new, {skip} skipped, {fail} failed, {no_url} no_url")


if __name__ == "__main__":
    main()

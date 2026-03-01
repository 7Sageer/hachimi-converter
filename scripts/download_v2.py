#!/usr/bin/env python3
"""Download paired audio: hachimi from API, original from YouTube search."""

import argparse
import json
import random
import subprocess
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PAIRS_FILE = Path(__file__).parent.parent / "pairs_full.jsonl"


def sanitize_filename(name):
    """Remove characters that are problematic in filenames."""
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        name = name.replace(ch, "_")
    return name[:80]


def download_hachimi(url, output_path, proxy=None):
    """Download hachimi audio directly from API URL."""
    if output_path.exists():
        return True
    try:
        handlers = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
        opener = urllib.request.build_opener(*handlers)
        req = urllib.request.Request(
            url,
            headers={
                "Referer": "https://hachimi.world/",
                "User-Agent": "HachimiWorld/1.0",
            },
        )
        with opener.open(req, timeout=30) as resp:
            output_path.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"    [fail] hachimi: {e}")
        return False


def download_original(query, output_path, proxy=None, max_retries=3):
    """Download original song from YouTube search, with retry."""
    if output_path.exists():
        return True
    cmd = [
        "yt-dlp",
        *(["--proxy", proxy] if proxy else []),
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--max-downloads",
        "1",
        "-o",
        str(output_path.with_suffix("")) + ".%(ext)s",
        f"ytsearch1:{query}",
    ]
    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            return True
        except subprocess.CalledProcessError as e:
            print(
                f"    [fail] original: exit code {e.returncode} (attempt {attempt}/{max_retries})"
            )
            if e.stderr:
                lines = e.stderr.strip().split("\n")
                for line in lines[-3:]:
                    print(f"      {line}")
            if attempt < max_retries:
                wait = 30 * attempt  # 30s, 60s
                print(f"    [retry] 等待 {wait}s 后重试...")
                time.sleep(wait)
        except Exception as e:
            print(f"    [fail] original: {e} (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(30 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description="Download paired audio: hachimi from API, original from YouTube")
    parser.add_argument("--proxy", type=str, default=None,
                        help="HTTP proxy URL (例如 http://localhost:10808)")
    args = parser.parse_args()

    (DATA_DIR / "hachimi").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "original").mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in PAIRS_FILE.read_text().strip().split("\n")]
    print(f"Total pairs: {len(pairs)}")

    ok_count = 0
    fail_streak = 0  # 连续失败计数

    skip_count = 0

    for i, p in enumerate(pairs):
        name = sanitize_filename(p["name"])
        hach_path = DATA_DIR / "hachimi" / f"{name}.mp3"
        orig_path = DATA_DIR / "original" / f"{name}.wav"

        # 两个文件都已存在，跳过
        if hach_path.exists() and orig_path.exists():
            skip_count += 1
            ok_count += 1
            continue

        print(f"\n[{i + 1}/{len(pairs)}] {p['name']}")

        # Download hachimi version
        h_ok = download_hachimi(p["hachimi_audio_url"], hach_path, proxy=args.proxy)
        if h_ok:
            print(f"    [ok] hachimi")

        # Download original
        query = p.get("youtube_query") or p.get("original_query") or p["original"]
        o_ok = download_original(query, orig_path, proxy=args.proxy)
        if o_ok:
            print(f"    [ok] original")
            fail_streak = 0
        else:
            fail_streak += 1

        if h_ok and o_ok:
            ok_count += 1

        # 连续失败 5 次，暂停 60 秒等风控解除
        if fail_streak >= 5:
            print(f"\n    [!] 连续失败 {fail_streak} 次，暂停 60 秒...")
            time.sleep(60)
            fail_streak = 0
        else:
            # 随机间隔 0.5-1.5 秒，降低触发风控概率
            delay = random.uniform(0.5, 1.5)
            time.sleep(delay)

    print(f"\nDone. {ok_count}/{len(pairs)} pairs ok (skipped {skip_count} already downloaded).")


if __name__ == "__main__":
    main()

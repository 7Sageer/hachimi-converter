#!/usr/bin/env python3
"""Download paired original/hachimi audio from Bilibili."""

import json
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PROXY = "http://localhost:10808"


def download_bilibili(bv: str, output_path: Path):
    """Download audio from a Bilibili video."""
    if output_path.exists():
        print(f"  [skip] {output_path.name} already exists")
        return True

    url = f"https://www.bilibili.com/video/{bv}/"
    cmd = [
        "yt-dlp",
        "--proxy", PROXY,
        "-x",  # extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        print(f"  [ok] {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [fail] {output_path.name}: {e.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [timeout] {output_path.name}")
        return False


def search_and_download_original(query: str, output_path: Path):
    """Search Bilibili for original song and download."""
    if output_path.exists():
        print(f"  [skip] {output_path.name} already exists")
        return True

    # Search bilibili via yt-dlp
    search_url = f"ytsearch1:{query}"
    cmd = [
        "yt-dlp",
        "--proxy", PROXY,
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
        search_url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        print(f"  [ok] {output_path.name}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  [fail] original search for '{query}'")
        return False


def main():
    pairs_file = DATA_DIR / "pairs.jsonl"
    if not pairs_file.exists():
        print("No pairs.jsonl found")
        sys.exit(1)

    pairs = [json.loads(line) for line in pairs_file.read_text().strip().split("\n")]

    for i, pair in enumerate(pairs):
        name = pair["name"]
        print(f"\n[{i+1}/{len(pairs)}] {name}")

        # Download hachimi version
        hachimi_path = DATA_DIR / "hachimi" / f"{name}.wav"
        print(f"  Downloading hachimi version...")
        download_bilibili(pair["hachimi_bv"], hachimi_path)

        # Download/search original
        original_path = DATA_DIR / "original" / f"{name}.wav"
        print(f"  Downloading original...")
        search_and_download_original(pair["original_query"], original_path)


if __name__ == "__main__":
    main()

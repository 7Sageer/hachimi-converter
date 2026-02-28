#!/usr/bin/env python3
"""Fetch all hachimi songs from Hachimi World API and build pairs.jsonl."""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

API_BASE = "https://api.hachimi.world"
PROXY = "http://localhost:10808"
HEADERS = {
    "Referer": "https://hachimi.world/",
    "X-Real-IP": "127.0.0.1",
    "User-Agent": "HachimiWorld/1.0",
}
LIMIT = 50
OUTPUT = Path(__file__).parent.parent / "pairs_full.jsonl"


def api_search(query="*", limit=LIMIT, offset=0):
    params = urllib.parse.urlencode({"q": query, "limit": limit, "offset": offset})
    url = f"{API_BASE}/song/search?{params}"
    
    proxy = urllib.request.ProxyHandler({"https": PROXY, "http": PROXY})
    opener = urllib.request.build_opener(proxy)
    req = urllib.request.Request(url, headers=HEADERS)
    
    with opener.open(req, timeout=15) as resp:
        data = json.loads(resp.read())
    
    if not data.get("ok"):
        raise Exception(f"API error: {data}")
    return data["data"]


def fetch_all():
    all_songs = []
    offset = 0
    
    while True:
        print(f"Fetching offset={offset}...")
        data = api_search(query="*", limit=LIMIT, offset=offset)
        hits = data["hits"]
        
        if not hits:
            break
        
        all_songs.extend(hits)
        offset += LIMIT
        print(f"  got {len(hits)} songs (total: {len(all_songs)})")
        
        if len(hits) < LIMIT:
            break
        
        time.sleep(0.5)
    
    return all_songs


def build_pairs(songs):
    """Convert API songs to training pairs."""
    pairs = []
    
    for s in songs:
        orig_titles = s.get("original_titles", [])
        orig_artists = s.get("original_artists", [])
        
        if not orig_titles:
            continue  # skip songs without original info
        
        pair = {
            "name": s["title"],
            "hachimi_id": s["display_id"],
            "hachimi_audio_url": s["audio_url"],
            "original": orig_titles[0],
            "original_artist": orig_artists[0] if orig_artists else "",
            "original_query": f"{orig_titles[0]} {orig_artists[0]}" if orig_artists else orig_titles[0],
            "youtube_query": f"{orig_titles[0]} {orig_artists[0]}" if orig_artists else orig_titles[0],
            "duration_seconds": s["duration_seconds"],
        }
        pairs.append(pair)
    
    return pairs


def main():
    print("Fetching all songs from Hachimi World...")
    songs = fetch_all()
    print(f"\nTotal songs fetched: {len(songs)}")
    
    pairs = build_pairs(songs)
    print(f"Songs with original info: {len(pairs)}")
    
    # Write jsonl
    with open(OUTPUT, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"Written to {OUTPUT}")
    
    # Stats
    with_artist = sum(1 for p in pairs if p["original_artist"])
    print(f"  With artist info: {with_artist}")
    print(f"  Without artist: {len(pairs) - with_artist}")


if __name__ == "__main__":
    main()

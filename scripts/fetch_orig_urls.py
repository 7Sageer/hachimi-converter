#!/usr/bin/env python3
"""Fetch detail for all pairs and extract original song URLs."""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

API_BASE = "https://api.hachimi.world"
PROXY = "http://localhost:10808"
HEADERS = {"Referer": "https://hachimi.world/", "User-Agent": "HachimiWorld/1.0"}
PAIRS_FILE = Path(__file__).parent.parent / "pairs_full.jsonl"
OUT_FILE = Path(__file__).parent.parent / "pairs_with_urls.jsonl"


def fetch_detail(display_id):
    params = urllib.parse.urlencode({"id": display_id})
    url = f"{API_BASE}/song/detail?{params}"
    proxy = urllib.request.ProxyHandler({"https": PROXY, "http": PROXY})
    opener = urllib.request.build_opener(proxy)
    req = urllib.request.Request(url, headers=HEADERS)
    with opener.open(req, timeout=15) as resp:
        return json.loads(resp.read())["data"]


def extract_orig_url(detail):
    """Extract best original song URL from detail."""
    # Prefer origin_infos[0].url
    for info in detail.get("origin_infos", []):
        url = info.get("url")
        if url and url.strip():
            return url.strip()
    # Fallback: bilibili external link
    for link in detail.get("external_links", []):
        if link["platform"] == "bilibili":
            return link["url"]
    return None


def main():
    pairs = [json.loads(l) for l in PAIRS_FILE.read_text().strip().split("\n")]
    print(f"Fetching details for {len(pairs)} pairs...")

    results = []
    for i, p in enumerate(pairs):
        try:
            detail = fetch_detail(p["hachimi_id"])
            orig_url = extract_orig_url(detail)
            p["orig_url"] = orig_url
            results.append(p)
            if orig_url:
                print(f"[{i+1}] ✅ {p['name'][:30]} → {orig_url[:60]}")
            else:
                print(f"[{i+1}] ⚠️  {p['name'][:30]} → no url")
        except Exception as e:
            p["orig_url"] = None
            results.append(p)
            print(f"[{i+1}] ❌ {p['name'][:30]}: {e}")
        time.sleep(0.3)

    with open(OUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    has_url = sum(1 for r in results if r.get("orig_url"))
    print(f"\nDone. {has_url}/{len(results)} have original URL → {OUT_FILE}")


if __name__ == "__main__":
    main()

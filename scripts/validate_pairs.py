#!/usr/bin/env python3
"""Validate paired audio quality using chroma cross-correlation.
Filters out mismatched pairs before training."""

import json
import librosa
import numpy as np
from pathlib import Path
from align_v2 import compute_normalized_chroma, search_best_offset

DATA_DIR = Path(__file__).parent.parent / "data"
PAIRS_FILE = Path(__file__).parent.parent / "pairs_full.jsonl"
SR = 22050
HOP = 512
MIN_SCORE = 0.5  # pairs below this are likely mismatched


def coarse_to_fine_best_score(chroma_ref, chroma_query):
    """Find best score with coarse scan + local refinement."""
    query_len = chroma_query.shape[1]
    ref_len = chroma_ref.shape[1]
    max_offset = ref_len - query_len
    if max_offset < 0:
        raise ValueError("query is longer than reference")
    if max_offset == 0:
        score = float(np.sum(chroma_ref * chroma_query) / query_len)
        return 0, score

    coarse_step = max(1, query_len // 20)
    coarse_offset, _ = search_best_offset(
        chroma_ref, chroma_query, start=0, end=max_offset, step=coarse_step
    )
    start = max(0, coarse_offset - coarse_step)
    end = min(max_offset, coarse_offset + coarse_step)
    return search_best_offset(chroma_ref, chroma_query, start=start, end=end, step=1)


def find_hachimi_path(safe_name):
    """Resolve hachimi audio path supporting both wav and mp3."""
    wav_path = DATA_DIR / "hachimi" / f"{safe_name}.wav"
    mp3_path = DATA_DIR / "hachimi" / f"{safe_name}.mp3"
    if wav_path.exists():
        return wav_path
    if mp3_path.exists():
        return mp3_path
    return None


def chroma_align_score(orig_path, hach_path):
    """Return best alignment score between original and hachimi."""
    try:
        y_orig, _ = librosa.load(orig_path, sr=SR, duration=120)
        y_hach, _ = librosa.load(hach_path, sr=SR, duration=120)
    except Exception as e:
        return 0.0, str(e)

    chroma_orig = compute_normalized_chroma(y_orig, sr=SR, hop_length=HOP)
    chroma_hach = compute_normalized_chroma(y_hach, sr=SR, hop_length=HOP)

    hach_len = chroma_hach.shape[1]
    orig_len = chroma_orig.shape[1]
    if hach_len == 0 or orig_len == 0:
        return 0.0, "empty_audio"

    if hach_len >= orig_len:
        _, score = coarse_to_fine_best_score(chroma_hach, chroma_orig)
        return float(score), "hachimi>=original"

    _, score = coarse_to_fine_best_score(chroma_orig, chroma_hach)
    return float(score), "ok"


def main():
    pairs = [json.loads(l) for l in PAIRS_FILE.read_text().strip().split("\n")]

    results = []
    for i, p in enumerate(pairs):
        name = p["name"]
        # sanitize filename same as download_v2
        safe = name
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            safe = safe.replace(ch, '_')
        safe = safe[:80]

        orig_path = DATA_DIR / "original" / f"{safe}.wav"
        hach_path = find_hachimi_path(safe)

        if not orig_path.exists() or hach_path is None:
            print(f"[{i+1}] {name}: MISSING FILES")
            continue

        score, note = chroma_align_score(orig_path, hach_path)
        status = "✅" if score >= MIN_SCORE else "❌"
        print(f"[{i+1}] {status} {score:.3f}  {name}  ← {p.get('original','?')}")
        results.append(
            {
                **p,
                "align_score": round(score, 3),
                "align_note": note,
                "valid": score >= MIN_SCORE,
            }
        )

    valid = [r for r in results if r["valid"]]
    invalid = [r for r in results if not r["valid"]]
    print(f"\nValid: {len(valid)}, Invalid: {len(invalid)}, Missing: {len(pairs)-len(results)}")

    # Write validated pairs
    out = Path(__file__).parent.parent / "pairs_validated.jsonl"
    with open(out, "w") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Validated pairs written to {out}")

    if invalid:
        print("\nRejected pairs:")
        for r in sorted(invalid, key=lambda x: x["align_score"]):
            print(f"  {r['align_score']:.3f}  {r['name']}  ← {r.get('original','?')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate paired audio quality using chroma cross-correlation.
Filters out mismatched pairs before training."""

import json
import librosa
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PAIRS_FILE = Path(__file__).parent.parent / "pairs_full.jsonl"
SR = 22050
HOP = 512
MIN_SCORE = 0.5  # pairs below this are likely mismatched


def chroma_align_score(orig_path, hach_path):
    """Return best alignment score between original and hachimi."""
    try:
        y_orig, _ = librosa.load(orig_path, sr=SR, duration=120)
        y_hach, _ = librosa.load(hach_path, sr=SR, duration=120)
    except Exception as e:
        return 0.0, str(e)

    chroma_orig = librosa.feature.chroma_cqt(y=y_orig, sr=SR, hop_length=HOP)
    chroma_hach = librosa.feature.chroma_cqt(y=y_hach, sr=SR, hop_length=HOP)

    # Normalize
    chroma_orig /= (np.linalg.norm(chroma_orig, axis=0, keepdims=True) + 1e-8)
    chroma_hach /= (np.linalg.norm(chroma_hach, axis=0, keepdims=True) + 1e-8)

    hach_len = chroma_hach.shape[1]
    orig_len = chroma_orig.shape[1]

    if hach_len >= orig_len:
        score = np.sum(chroma_orig * chroma_hach[:, :orig_len]) / orig_len
        return float(score), "hachimi>=original"

    best = -1.0
    step = max(1, hach_len // 4)  # faster scan
    for offset in range(0, orig_len - hach_len + 1, step):
        window = chroma_orig[:, offset:offset + hach_len]
        score = float(np.sum(window * chroma_hach) / hach_len)
        if score > best:
            best = score

    return best, "ok"


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
        hach_path = DATA_DIR / "hachimi" / f"{safe}.mp3"

        if not orig_path.exists() or not hach_path.exists():
            print(f"[{i+1}] {name}: MISSING FILES")
            continue

        score, note = chroma_align_score(orig_path, hach_path)
        status = "✅" if score >= MIN_SCORE else "❌"
        print(f"[{i+1}] {status} {score:.3f}  {name}  ← {p.get('original','?')}")
        results.append({**p, "align_score": round(score, 3), "valid": score >= MIN_SCORE})

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

#!/usr/bin/env python3
"""Slice paired audio using chroma-based offset alignment."""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from align_v2 import compute_normalized_chroma, find_alignment, refine_local_offset

DATA_DIR = Path(__file__).parent.parent / "data"
PAIRED_DIR = DATA_DIR / "paired"
SR = 22050
ALIGN_HOP = 512  # hop for chroma alignment — independent of mel pipeline
SEGMENT_SEC = 3
LOCAL_SEARCH_SEC = 0.35
LOCAL_MIN_SCORE = 0.35


def slice_pair(name, orig_path, hach_path):
    """Find alignment offset, then slice both into paired segments."""
    y_orig, _ = librosa.load(orig_path, sr=SR)
    y_hach, _ = librosa.load(hach_path, sr=SR)

    # Find where hachimi aligns in original
    offset_frames, _, score = find_alignment(orig_path, hach_path)
    if score < 0.3:
        print(f"  [skip] score too low: {score:.3f}")
        return 0

    chroma_orig = compute_normalized_chroma(y_orig)
    chroma_hach = compute_normalized_chroma(y_hach)

    seg_samples = SEGMENT_SEC * SR
    seg_frames = max(1, int(round(seg_samples / ALIGN_HOP)))
    search_radius_frames = max(1, int(round(LOCAL_SEARCH_SEC * SR / ALIGN_HOP)))
    n_segments = len(y_hach) // seg_samples

    count = 0
    local_scores = []
    skipped_local = 0
    for i in range(n_segments):
        hach_start = i * seg_samples
        hach_end = hach_start + seg_samples

        hach_frame_start = int(round(hach_start / ALIGN_HOP))
        expected_orig_frame = offset_frames + hach_frame_start
        local_orig_frame, local_score = refine_local_offset(
            chroma_orig=chroma_orig,
            chroma_hach=chroma_hach,
            hach_start_frame=hach_frame_start,
            seg_frames=seg_frames,
            expected_orig_frame=expected_orig_frame,
            search_radius_frames=search_radius_frames,
        )
        if local_score < LOCAL_MIN_SCORE:
            skipped_local += 1
            continue

        orig_start = local_orig_frame * ALIGN_HOP
        orig_end = orig_start + seg_samples

        if hach_end > len(y_hach) or orig_end > len(y_orig):
            break

        sf.write(PAIRED_DIR / f"{name}_orig_{count:03d}.wav",
                 y_orig[orig_start:orig_end], SR)
        sf.write(PAIRED_DIR / f"{name}_hach_{count:03d}.wav",
                 y_hach[hach_start:hach_end], SR)
        count += 1
        local_scores.append(local_score)

    if local_scores:
        print(
            f"  local align score avg={np.mean(local_scores):.3f}, "
            f"min={np.min(local_scores):.3f}, skipped={skipped_local}"
        )
    elif skipped_local:
        print("  [skip] all segments rejected by local alignment threshold")

    return count


def main():
    import shutil
    # Clean old paired data
    if PAIRED_DIR.exists():
        shutil.rmtree(PAIRED_DIR)
    PAIRED_DIR.mkdir(parents=True)

    # Support both .wav and .mp3 hachimi files
    names = [p.stem for p in (DATA_DIR / "hachimi").glob("*.wav")]
    names += [p.stem for p in (DATA_DIR / "hachimi").glob("*.mp3")]
    names = sorted(set(names))
    total = 0

    for name in sorted(names):
        orig_path = DATA_DIR / "original" / f"{name}.wav"
        # Support both .wav and .mp3 hachimi files
        hach_path = DATA_DIR / "hachimi" / f"{name}.wav"
        if not hach_path.exists():
            hach_path = DATA_DIR / "hachimi" / f"{name}.mp3"
        if not orig_path.exists() or not hach_path.exists():
            continue

        print(f"\n[{name}]")
        n = slice_pair(name, orig_path, hach_path)
        print(f"  → {n} segments")
        total += n

    print(f"\nTotal: {total} paired segments in {PAIRED_DIR}/")


if __name__ == "__main__":
    main()

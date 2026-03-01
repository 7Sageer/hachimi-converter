#!/usr/bin/env python3
"""Find where hachimi version aligns within the original song using chroma cross-correlation."""

import librosa
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SR = 22050
ALIGN_HOP = 512  # hop for chroma alignment — independent of mel pipeline


def compute_normalized_chroma(y, sr=SR, hop_length=ALIGN_HOP):
    """Compute column-normalized chroma features."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    return chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)


def _score_window(chroma_ref, chroma_query, offset):
    """Cosine-style mean similarity for a query window at `offset`."""
    q_len = chroma_query.shape[1]
    window = chroma_ref[:, offset:offset + q_len]
    return float(np.sum(window * chroma_query) / q_len)


def search_best_offset(chroma_ref, chroma_query, start=0, end=None, step=1):
    """Search best offset where `chroma_query` matches inside `chroma_ref`."""
    q_len = chroma_query.shape[1]
    r_len = chroma_ref.shape[1]
    max_start = r_len - q_len
    if max_start < 0:
        raise ValueError("query is longer than reference")

    step = max(1, int(step))
    start = max(0, int(start))
    end = max_start if end is None else min(max_start, int(end))
    if start > end:
        start = end

    best_offset = start
    best_score = -1.0
    for offset in range(start, end + 1, step):
        score = _score_window(chroma_ref, chroma_query, offset)
        if score > best_score:
            best_score = score
            best_offset = offset

    if (end - start) % step != 0:
        score = _score_window(chroma_ref, chroma_query, end)
        if score > best_score:
            best_score = score
            best_offset = end

    return best_offset, best_score


def refine_local_offset(
    chroma_orig,
    chroma_hach,
    hach_start_frame,
    seg_frames,
    expected_orig_frame,
    search_radius_frames,
):
    """Refine local segment offset near expected location."""
    if seg_frames <= 0:
        return 0, 0.0

    hach_start_frame = int(hach_start_frame)
    seg_frames = int(seg_frames)
    hach_end = hach_start_frame + seg_frames
    if hach_start_frame < 0 or hach_end > chroma_hach.shape[1]:
        return 0, 0.0

    query = chroma_hach[:, hach_start_frame:hach_end]
    max_start = chroma_orig.shape[1] - seg_frames
    if max_start < 0:
        return 0, 0.0

    expected_orig_frame = int(np.clip(expected_orig_frame, 0, max_start))
    radius = max(0, int(search_radius_frames))
    start = max(0, expected_orig_frame - radius)
    end = min(max_start, expected_orig_frame + radius)

    return search_best_offset(chroma_orig, query, start=start, end=end, step=1)


def find_alignment(orig_path, hach_path):
    """Find the best offset of hachimi within original using chroma cross-correlation."""
    y_orig, _ = librosa.load(orig_path, sr=SR)
    y_hach, _ = librosa.load(hach_path, sr=SR)

    print(f"  orig: {len(y_orig)/SR:.1f}s, hach: {len(y_hach)/SR:.1f}s")

    chroma_orig = compute_normalized_chroma(y_orig)
    chroma_hach = compute_normalized_chroma(y_hach)

    # Sliding window cross-correlation
    hach_len = chroma_hach.shape[1]
    orig_len = chroma_orig.shape[1]
    if hach_len == 0 or orig_len == 0:
        return 0, 0, 0.0

    if hach_len >= orig_len:
        # Do not force score=1.0; evaluate best overlap of original within hachimi.
        _, best_score = search_best_offset(chroma_hach, chroma_orig, step=1)
        print("  Hachimi is longer than original; using best overlap score")
        print(f"  Best overlap score={best_score:.3f}")
        return 0, orig_len, best_score

    # Two-stage search: coarse scan + local refinement.
    coarse_step = max(1, hach_len // 20)
    coarse_offset, _ = search_best_offset(
        chroma_orig,
        chroma_hach,
        start=0,
        end=orig_len - hach_len,
        step=coarse_step,
    )
    search_start = max(0, coarse_offset - coarse_step)
    search_end = min(orig_len - hach_len, coarse_offset + coarse_step)
    best_offset, best_score = search_best_offset(
        chroma_orig, chroma_hach, start=search_start, end=search_end, step=1
    )

    offset_sec = best_offset * ALIGN_HOP / SR
    duration_sec = hach_len * ALIGN_HOP / SR
    print(f"  Best match: offset={offset_sec:.1f}s, duration={duration_sec:.1f}s, score={best_score:.3f}")

    return best_offset, hach_len, best_score


def main():
    names = [p.stem for p in (DATA_DIR / "hachimi").glob("*.wav")]

    for name in sorted(names):
        orig_path = DATA_DIR / "original" / f"{name}.wav"
        hach_path = DATA_DIR / "hachimi" / f"{name}.wav"

        if not orig_path.exists() or not hach_path.exists():
            continue

        print(f"\n[{name}]")
        offset, length, score = find_alignment(orig_path, hach_path)

        if score < 0.3:
            print(f"  ⚠️ LOW SCORE — probably wrong original or heavily rearranged")
        elif score < 0.5:
            print(f"  ⚠️ Mediocre alignment — may need manual check")
        else:
            print(f"  ✅ Good alignment")


if __name__ == "__main__":
    main()

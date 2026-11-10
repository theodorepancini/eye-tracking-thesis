"""
A1 - Reproduce Lu et al. (2026) CGF results on the full LPW frame set.

Runs the reference CGF detector (xuepx/cgf_pupil_detector @ 0010408) on every
frame of every LPW video and compares the detected pupil center with the LPW
annotation. Metrics follow Lu et al. Table 3 / Section 5.3:

  - detection rate: found / ALL frames
  - center error (median, mean, within 5/10 px): only frames where the method
    found a pupil AND an annotation exists
  - RITNet rate: share of frames where the fallback was invoked

Usage (from the eye-tracking-thesis folder):
  python experiments/a1_reproduce/evaluate_lpw.py
  python experiments/a1_reproduce/evaluate_lpw.py --participants 1 --max-frames 200   # quick test
"""
import argparse   # reads command-line options like --threshold
import csv        # writes the per-frame results file
import json       # writes the summary file
import math       # math.isnan() / math.nan for missing values
import sys        # sys.path, so Python can find the reference repo
import time       # measures total run time

import cv2                 # reads the .avi videos and converts to grayscale
import numpy         # median/mean over all errors
from pathlib import Path   # handles file paths the same on Windows and Linux
from tqdm import tqdm      # progress bar

# --- Paths -------------------------------------------------------------------
# All paths are built from where THIS file lives, so the script works no matter
# which folder you run it from.
SCRIPT_DIR = Path(__file__).resolve().parent          # .../experiments/a1_reproduce
REPO_DIR = SCRIPT_DIR.parents[1]                      # .../eye-tracking-thesis (two levels up)
CGF_DIR = REPO_DIR.parent / "cgf_pupil_detector"      # their repo, cloned next to ours

# Their repo is not an installed package, so we add its folder to Python's
# search path. Their files import each other by bare name (e.g. "import config"),
# which is why the folder itself has to be on the path.
sys.path.insert(0, str(CGF_DIR))

# Reuse their own code, unchanged, so we measure exactly what they published.
from cgf_detector import CGFDetector                  # noqa: E402  the detector (traditional + RITNet fallback)
from demo import load_lpw_annotations                 # noqa: E402  reads the LPW .txt (one "x y" per frame)
from metrics import compute_center_error              # noqa: E402  Euclidean distance in pixels


def find_videos(lpw_dir, participants=None):
    """Return sorted list of (participant, video_id, avi_path, txt_path).

    LPW layout is LPW/<participant>/<video>.avi with a matching .txt annotation.
    """
    # Keep only numbered participant folders (skips stray files/folders),
    # then sort them by number: 1, 2, ..., 22 (not 1, 10, 11, ...)
    participant_dirs = [p for p in lpw_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    participant_dirs.sort(key=lambda p: int(p.name))

    videos = []
    for pdir in participant_dirs:
        participant = int(pdir.name)
        if participants and participant not in participants:
            continue                                  # --participants filter

        # Sort videos by number too: stem "2" -> "0002" sorts before "10" -> "0010"
        for avi in sorted(pdir.glob("*.avi"), key=lambda p: p.stem.zfill(4)):
            txt = avi.with_suffix(".txt")             # annotation has the same name as the video
            videos.append((participant, avi.stem, avi, txt))
    return videos


def summarize(rows):
    """Compute the paper's metrics from the per-frame rows.

    Matches Lu et al. Table 3: detection rate uses ALL frames as denominator;
    error metrics use only frames where a pupil was found and a GT exists.
    """
    total = len(rows)
    frames_found = 0
    ritnet_count = 0
    errors = []

    for r in rows:
        if r["found"]:
            frames_found += 1
            # Error is NaN when there was no annotation (see main())
            if not math.isnan(r["error"]):
                errors.append(r["error"])
        # RITNet fallback invoked (found or not)
        if r["method"] == "RITNet":
            ritnet_count += 1

    # Default to NaN; overwrite only if there is data to divide by
    detection_rate = ritnet_rate = math.nan
    if total > 0:
        detection_rate = frames_found / total
        ritnet_rate = ritnet_count / total

    median_error = mean_error = within_5 = within_10 = math.nan
    if errors:
        median_error = float(numpy.median(errors))
        mean_error = float(numpy.mean(errors))
        within_5 = sum(e <= 5 for e in errors) / len(errors)    # share <= 5 px
        within_10 = sum(e <= 10 for e in errors) / len(errors)  # share <= 10 px

    return {
        "frames_total": total,
        "frames_found": frames_found,
        "frames_with_error": len(errors),
        "detection_rate": detection_rate,
        "ritnet_rate": ritnet_rate,
        "median_error_px": median_error,
        "mean_error_px": mean_error,
        "within_5px": within_5,
        "within_10px": within_10,
    }


def main():
    # --- Command-line options (all have defaults, so no arguments are required)
    ap = argparse.ArgumentParser(description="Evaluate reference CGF on LPW")
    ap.add_argument("--lpw-dir", type=Path, default=REPO_DIR / "data" / "LPW")
    ap.add_argument("--weights", type=Path,
                    default=CGF_DIR / "best_model_finetune_v2_ep122_miou9524.pkl")   # their RITNet weights
    ap.add_argument("--threshold", type=float, default=0.8, help="tau (paper: 0.8 accuracy mode)")
    ap.add_argument("--participants", type=int, nargs="*", help="only these participant IDs")
    ap.add_argument("--max-frames", type=int, default=None, help="limit frames per video (testing)")
    ap.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "a1_reproduce")
    args = ap.parse_args()

    videos = find_videos(args.lpw_dir, args.participants)
    if not videos:
        sys.exit(f"No videos found in {args.lpw_dir}")

    # Output files are named after tau, so runs with different thresholds
    # don't overwrite each other.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"tau{args.threshold}"
    csv_path = args.out_dir / f"per_frame_{tag}.csv"
    json_path = args.out_dir / f"summary_{tag}.json"

    # Create the detector ONCE. RITNet is only loaded the first time it is
    # needed (lazy loading inside their class).
    detector = CGFDetector(weights_path=str(args.weights), threshold=args.threshold)

    rows = []       # one dict per frame -> becomes the CSV
    warnings = []   # videos where frame count != annotation count
    t0 = time.perf_counter()

    for participant, vid, avi, txt in tqdm(videos, desc="Videos"):
        # Load all annotations for this video once (list of (x, y), one per frame)
        gt = []
        if txt.exists():
            gt = load_lpw_annotations(str(txt))

        cap = cv2.VideoCapture(str(avi))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # If these differ, frame i might not match annotation i -> flag it
        if len(gt) != n_video:
            warnings.append(f"P{participant}/{vid}: {n_video} frames but {len(gt)} annotations")

        i = 0   # frame index within this video
        while args.max_frames is None or i < args.max_frames:   # --max-frames limit (testing)
            ok, frame = cap.read()                    # read next frame
            if not ok:
                break                                 # end of video

            # LPW frames are read as 3-channel BGR; the detector expects grayscale
            # (same conversion as their demo.py)
            gray = frame
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            res = detector.detect(gray)               # the actual CGF detection

            # Ground-truth center for this frame (None if the .txt is too short)
            gt_c = None
            if i < len(gt):
                gt_c = gt[i]

            # Error only makes sense if a pupil was found; otherwise NaN
            # (compute_center_error also returns NaN if gt_c is None)
            err = math.nan
            if res["found"]:
                err = compute_center_error(res["center"], gt_c)

            # Default every coordinate to NaN, fill in what exists
            cx = cy = gt_x = gt_y = math.nan
            if res["center"] is not None:
                cx, cy = res["center"]
            if gt_c:
                gt_x, gt_y = gt_c[0], gt_c[1]

            # Save everything per frame, so metrics can be recomputed later
            # without re-running the whole dataset
            rows.append({
                "participant": participant, "video": vid, "frame": i,
                "found": bool(res["found"]),
                "method": res["method"],              # "traditional" or "RITNet"
                "confidence": res["confidence"],      # geometric score of the traditional result
                "pred_x": cx, "pred_y": cy,
                "gt_x": gt_x, "gt_y": gt_y,
                "diameter": res["diameter"],          # mean of ellipse axes (not evaluated by Lu et al.)
                "error": err,
            })
            i += 1
        cap.release()

    elapsed = time.perf_counter() - t0

    # --- Write per-frame results
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- Write summary, plus everything needed to know HOW the result was produced
    summary = summarize(rows)
    summary.update({
        "threshold": args.threshold,
        "videos": len(videos),
        "wall_time_s": elapsed,
        "note": "wall time includes video decoding; not comparable to paper FPS",
        "reference_commit": "xuepx/cgf_pupil_detector@0010408",
        "opencv": cv2.__version__,
        "warnings": warnings,
    })
    try:
        import torch
        summary["torch"] = torch.__version__
    except ImportError:
        pass                                          # torch version is optional info
    json_path.write_text(json.dumps(summary, indent=2))

    # --- Print results next to the paper's numbers for quick comparison
    print("\n--- A1 summary ---")
    for k in ["frames_total", "detection_rate", "ritnet_rate", "median_error_px",
              "mean_error_px", "within_5px", "within_10px"]:
        value = summary[k]
        if isinstance(value, float):
            print(f"  {k:16s} {value:.4f}")           # rates/errors: 4 decimals
        else:
            print(f"  {k:16s} {value}")               # counts: as-is
    print("  Paper/README full set (tau=0.8): median 1.40 px, within 5px 86.8%, det. rate 74.8%")
    if warnings:
        print(f"  {len(warnings)} annotation/frame-count mismatches (see summary JSON)")
    print(f"\nSaved {csv_path}\nSaved {json_path}")


# Only run main() when the file is executed directly, not when imported
if __name__ == "__main__":
    main()
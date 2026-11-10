"""
B - Run the reference CGF on OUR OWN recordings.

Same detector as A1 (xuepx/cgf_pupil_detector @ 0010408), but:
  - runs on one or more of our own videos (not LPW)
  - can shrink the input before detection (--downscale), so their ORIGINAL
    parameters fit our much larger pupil (~285 px vs LPW's ~62 px)
  - can adjust their size limits / expected pupil size (--pupil-diameter),
    without editing their files
  - records the four confidence factors per frame, to see WHY frames fall
    back to RITNet

Findings 2026-11-06 (theo_fixed, cam0): --downscale 2 with their original
parameters worked best (79.8 % traditional, 18.9 % RITNet, 1.3 % not found).
Scaling block size/kernel and removing glints made it worse, so those
options were removed.

Annotations are optional: if a .txt with one "x y" per frame sits next to a
video (same name), center errors are computed; otherwise only detection rate,
RITNet rate and confidence factors are reported.

Usage (from the eye-tracking-thesis folder):
  python experiments/b_own_data/evaluate_own.py data/own/rec/cam0.avi                     # their original parameters
  python experiments/b_own_data/evaluate_own.py data/own/rec/cam0.avi --downscale 2       # best so far
  python experiments/b_own_data/evaluate_own.py data/own/rec/cam0.avi --pupil-diameter 285            # size limits + expected size
  python experiments/b_own_data/evaluate_own.py data/own/rec/cam0.avi --downscale 2 --pupil-diameter 141 --only prior

--pupil-diameter is in the pixels the detector sees (after --downscale).
Scaling: s = D / 62 px (62 px = diameter of LPW's expected pupil area,
3000 px^2); areas x s^2.
"""
import argparse
import csv
import json
import math
import sys
import time

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Paths (built from this file's location, works from any folder) ----------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../experiments/b_own_data
REPO_DIR = SCRIPT_DIR.parents[1]                      # .../eye-tracking-thesis
CGF_DIR = REPO_DIR.parent / "cgf_pupil_detector"      # their repo, next to ours
sys.path.insert(0, str(CGF_DIR))

from cgf_detector import CGFDetector                  # noqa: E402
from demo import load_lpw_annotations                 # noqa: E402  same "x y" per line format
from metrics import compute_center_error              # noqa: E402
import config                                         # noqa: E402  their parameters (read at run time)
import confidence_score                               # noqa: E402  holds EXPECTED_AREA_LOG_MU

LPW_PUPIL_DIAMETER = 62.0   # px; circle with LPW's expected area 3000 px^2
FACTORS = ["circularity", "aspect_ratio", "area_score", "convex_ratio"]


def scale_parameters(diameter, only="area"):
    """Override their size-dependent parameters IN MEMORY for this run.

    Their files are not edited. Must be called before the detector is created.
    Block size and morph kernel are never changed: scaling them made detection
    worse (the iris and shadows merged with the pupil).
      "area"  = size limits + expected pupil size (default)
      "prior" = only the expected pupil size in the confidence score;
                thresholding and candidate filtering stay exactly as in the original
    """
    s = diameter / LPW_PUPIL_DIAMETER
    if only == "area":
        config.MIN_PUPIL_AREA = 100 * s ** 2
        config.MAX_PUPIL_AREA = 50000 * s ** 2
    confidence_score.EXPECTED_AREA_LOG_MU = np.log(3000.0 * s ** 2)
    return s


def detect_downscaled(detector, gray, factor):
    """Run THEIR detector on an image shrunk by `factor`, then scale the result back.

    Our pupil (~285 px) is much larger than LPW's (~62 px), so their fixed
    parameters (block size, kernel, size limits, expected area) don't fit.
    Shrinking the input brings the pupil-to-window/image proportions closer to
    LPW's, so their original parameters apply unchanged; only the input scale
    differs. Factor 2 worked best; 4 made it worse (window covers ~1/3 of the
    image, CLAHE tiles become tiny).
    """
    if factor == 1:
        return detector.detect(gray)

    h, w = gray.shape[:2]
    small = cv2.resize(gray, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    res = dict(detector.detect(small))                # copy, so we can change it

    # Scale every size/position back to the original image
    if res["center"] is not None:
        x, y = res["center"]
        res["center"] = (x * factor, y * factor)
    if res["ellipse"] is not None:
        (cx, cy), (a, b), ang = res["ellipse"]
        res["ellipse"] = ((cx * factor, cy * factor), (a * factor, b * factor), ang)
    res["diameter"] = res["diameter"] * factor          # NaN stays NaN
    if res.get("mask") is not None:
        res["mask"] = cv2.resize(res["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
    return res


def current_parameters(scale):
    """Values in effect for this run - saved with the results."""
    return {
        "scale": scale,
        "ADAPTIVE_BLOCK_SIZE": config.ADAPTIVE_BLOCK_SIZE,   # always their original (51)
        "MORPH_KERNEL_SIZE": config.MORPH_KERNEL_SIZE,       # always their original (5)
        "MIN_PUPIL_AREA": config.MIN_PUPIL_AREA,
        "MAX_PUPIL_AREA": config.MAX_PUPIL_AREA,
        "EXPECTED_AREA": float(np.exp(confidence_score.EXPECTED_AREA_LOG_MU)),
        "ADAPTIVE_C": config.ADAPTIVE_C,
    }


def summarize(rows):
    total = len(rows)
    found_count = 0
    ritnet_count = 0
    trad_found = []     # rows where the traditional branch produced a candidate
    confidences = []
    diameters = []
    errors = []

    for r in rows:
        if r["found"]:
            found_count += 1
            # Error is NaN when there was no annotation
            if not math.isnan(r["error"]):
                errors.append(r["error"])
        if r["method"] == "RITNet":
            ritnet_count += 1
        if r["f_convex_ratio"] > 0:
            trad_found.append(r)
        confidences.append(r["confidence"])
        diameters.append(r["diameter"])

    # Default to NaN; overwrite only if there is data
    detection_rate = ritnet_rate = trad_candidate_rate = math.nan
    if total > 0:
        detection_rate = found_count / total
        ritnet_rate = ritnet_count / total
        # How often the traditional branch found ANY candidate (before the tau check)
        trad_candidate_rate = len(trad_found) / total

    mean_confidence = median_diameter = math.nan
    if rows:
        mean_confidence = float(np.mean(confidences))
        median_diameter = float(np.nanmedian(diameters))   # ignores NaN diameters

    # Average of each factor where traditional found a candidate -> the low one is the culprit
    mean_factors = {}
    for k in FACTORS:
        values = [r[f"f_{k}"] for r in trad_found]
        mean_factors[k] = float(np.mean(values)) if values else math.nan

    median_error = within_5 = math.nan
    if errors:
        median_error = float(np.median(errors))
        within_5 = sum(e <= 5 for e in errors) / len(errors)    # share <= 5 px

    return {
        "frames_total": total,
        "detection_rate": detection_rate,
        "ritnet_rate": ritnet_rate,
        "traditional_candidate_rate": trad_candidate_rate,
        "mean_confidence": mean_confidence,
        "mean_factors": mean_factors,
        "median_diameter_px": median_diameter,
        "frames_with_error": len(errors),
        "median_error_px": median_error,
        "within_5px": within_5,
    }


def main():
    ap = argparse.ArgumentParser(description="Run reference CGF on our own recordings")
    ap.add_argument("videos", type=Path, nargs="+", help="one or more video files")
    ap.add_argument("--pupil-diameter", type=float, default=None,
                    help="our pupil diameter in px (as the detector sees it, i.e. after --downscale); "
                         "adjusts their size limits and expected pupil size")
    ap.add_argument("--only", choices=["area", "prior"], default="area",
                    help="with --pupil-diameter: 'area' = size limits + expected size (default), "
                         "'prior' = only the expected size in the confidence score")
    ap.add_argument("--threshold", type=float, default=0.8, help="tau (paper: 0.8)")
    ap.add_argument("--downscale", type=int, default=1,
                    help="shrink input by this factor before detection, results scaled back "
                         "(2 worked best on our recordings)")
    ap.add_argument("--max-frames", type=int, default=None, help="limit frames per video (testing)")
    ap.add_argument("--weights", type=Path,
                    default=CGF_DIR / "best_model_finetune_v2_ep122_miou9524.pkl")
    ap.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "own_data")
    args = ap.parse_args()

    for v in args.videos:
        if not v.exists():
            sys.exit(f"Video not found: {v}")

    # Scale BEFORE creating the detector
    s = scale_parameters(args.pupil_diameter, args.only) if args.pupil_diameter else 1.0
    params = current_parameters(s)
    detector = CGFDetector(weights_path=str(args.weights), threshold=args.threshold)

    # File names describe the run, so runs don't overwrite each other
    # Include the recording folder: every recording has a cam0.avi, so the stem alone would overwrite
    video = args.videos[0]
    name = f"{video.parent.name}_{video.stem}" if len(args.videos) == 1 else f"{len(args.videos)}videos"
    tag = f"{name}_tau{args.threshold}"
    if args.downscale > 1:
        tag += f"_ds{args.downscale}"
    if args.pupil_diameter:
        tag += f"_d{args.pupil_diameter:g}_{args.only}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"per_frame_{tag}.csv"
    json_path = args.out_dir / f"summary_{tag}.json"

    rows = []
    t0 = time.perf_counter()
    for video in args.videos:
        # Annotations are optional: same name as the video, but .txt
        txt = video.with_suffix(".txt")
        gt = load_lpw_annotations(str(txt)) if txt.exists() else []

        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.max_frames:
            n = min(n, args.max_frames)               # --max-frames limit (testing)

        for i in tqdm(range(n), desc=video.name):
            ok, frame = cap.read()
            if not ok:
                break                                 # end of video

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            res = detect_downscaled(detector, gray, args.downscale)

            # Ground-truth center for this frame (None if no/too short annotation)
            gt_c = gt[i] if i < len(gt) else None
            # Error only makes sense if a pupil was found; otherwise NaN
            err = compute_center_error(res["center"], gt_c) if res["found"] else math.nan
            cx, cy = res["center"] if res["center"] is not None else (math.nan, math.nan)
            gt_x, gt_y = (gt_c[0], gt_c[1]) if gt_c else (math.nan, math.nan)

            row = {
                "video": video.name, "frame": i,
                "found": bool(res["found"]), "method": res["method"],
                "confidence": res["confidence"],
            }
            # One column per confidence factor: f_circularity, f_aspect_ratio, ...
            for k in FACTORS:
                row[f"f_{k}"] = res["confidence_factors"][k]
            row.update({
                "pred_x": cx, "pred_y": cy, "diameter": res["diameter"],
                "gt_x": gt_x, "gt_y": gt_y,
                "error": err,
            })
            rows.append(row)
        cap.release()

    elapsed = time.perf_counter() - t0
    if not rows:
        sys.exit("No frames could be read")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    summary.update({
        "videos": [str(v) for v in args.videos],
        "threshold": args.threshold,
        "parameters": params,
        "downscale": args.downscale,
        "wall_time_s": elapsed,
        "reference_commit": "xuepx/cgf_pupil_detector@0010408",
        "opencv": cv2.__version__,
    })
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\n--- {tag} ---")
    print(f"  downscale {args.downscale}, block {params['ADAPTIVE_BLOCK_SIZE']}, kernel {params['MORPH_KERNEL_SIZE']}, "
          f"size limits {params['MIN_PUPIL_AREA']:.0f}-{params['MAX_PUPIL_AREA']:.0f} px^2, "
          f"expected area {params['EXPECTED_AREA']:.0f} px^2")
    for k in ["frames_total", "detection_rate", "ritnet_rate", "traditional_candidate_rate",
              "mean_confidence", "median_diameter_px", "median_error_px", "within_5px"]:
        value = summary[k]
        text = f"{value:.3f}" if isinstance(value, float) else value
        print(f"  {k:27s} {text}")
    print("  mean factors: " + ", ".join(f"{k} {v:.2f}" for k, v in summary["mean_factors"].items()))
    print(f"\nSaved {csv_path}\nSaved {json_path}")


if __name__ == "__main__":
    main()
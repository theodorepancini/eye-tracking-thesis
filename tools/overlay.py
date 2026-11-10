"""
Visual check of the reference CGF detection on our own recordings.

Draws the detected pupil ellipse and center on EVERY frame, so you can see
whether the detection is correct. No annotations needed.

Outputs (next to the video, or in --out-dir):
  <name>_overlay.avi   every frame with the detection drawn on it
                       (step frame by frame in VLC with the E key)
  <name>_sheet.png     a grid of evenly spaced frames for a quick spot check

Colours: green = traditional detection, red = RITNet fallback,
         yellow text "NOT FOUND" = no pupil detected.

Usage (from the eye-tracking-thesis folder):
  python tools/overlay.py data/own/20261106_theo_fixed/cam0.avi                    # their original parameters
  python tools/overlay.py data/own/20261106_theo_fixed/cam0.avi --downscale 2       # best so far
  python tools/overlay.py data/own/20261106_theo_fixed/cam0.avi --downscale 2 --show-mask --sheet 48
  python tools/overlay.py data/own/.../cam0.avi --downscale 2 --pupil-diameter 141 --only prior

--show-mask puts the traditional branch's binary mask next to each frame, with
every region coloured by the filter that kept or rejected it:
  green = candidate, red = circularity too low, blue = wrong area, orange = too elongated

This only shows WHERE the detector put the pupil. Measured accuracy needs
manual annotation (planned 11-12 Nov).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "experiments" / "b_own_data"))
# Reuses the same detector setup and parameter scaling as evaluate_own.py
from evaluate_own import CGFDetector, CGF_DIR, scale_parameters, detect_downscaled   # noqa: E402

import config                                                     # noqa: E402  their parameters (already scaled, same module)
from traditional_methods import _to_pupil_mask                    # noqa: E402  their morphology step

GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)
BLUE, ORANGE, GRAY = (255, 120, 0), (0, 140, 255), (128, 128, 128)


def aspect_ratio(cnt):
    """Long axis / short axis of the ellipse fitted to a contour (needs >= 5 points)."""
    _, axes, _ = cv2.fitEllipse(cnt)
    return max(axes) / max(min(axes), 1e-6)       # 1e-6 avoids dividing by zero


def mask_view(frame_gray, detector, factor=1):
    """Show the traditional branch's binary mask and why each region was kept or rejected.

    Repeats their steps exactly: preprocess (gamma + CLAHE) -> adaptive mean
    threshold -> morphology -> external contours -> the same filters as
    pupil_geometry.extract_pupil_from_mask. Contour colours:
      green  = passes all filters (a pupil candidate)
      red    = rejected: circularity below MIN_CIRCULARITY
      blue   = rejected: area outside MIN/MAX_PUPIL_AREA
      orange = rejected: ellipse aspect ratio above MAX_ASPECT_RATIO
    """
    full_h, full_w = frame_gray.shape[:2]
    if factor > 1:   # the mask is computed on the same shrunk image the detector sees
        frame_gray = cv2.resize(frame_gray, (full_w // factor, full_h // factor), interpolation=cv2.INTER_AREA)

    pre = detector.preprocess(frame_gray)
    binary = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                   blockSize=config.ADAPTIVE_BLOCK_SIZE, C=config.ADAPTIVE_C)
    mask = _to_pupil_mask(binary)                                  # 0/1 after open/close
    mask_u8 = (mask * 255).astype(np.uint8)                        # 0/255, what OpenCV expects
    vis = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR) // 3           # dim white = dark pixels
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if area < config.MIN_PUPIL_AREA / 4 or perim == 0:
            continue                                               # tiny specks: don't clutter the view
        circ = 4 * np.pi * area / perim ** 2                       # 1 = perfect circle

        # Same filter order as theirs: the first one that fails decides the colour
        if not (config.MIN_PUPIL_AREA <= area <= config.MAX_PUPIL_AREA):
            colour, why = BLUE, "area"
        elif circ < config.MIN_CIRCULARITY:
            colour, why = RED, "circ"
        elif len(cnt) >= 5 and aspect_ratio(cnt) > config.MAX_ASPECT_RATIO:
            colour, why = ORANGE, "aspect"
        else:
            colour, why = GREEN, "ok"

        cv2.drawContours(vis, [cnt], -1, colour, 2)
        x, y, _, _ = cv2.boundingRect(cnt)
        cv2.putText(vis, f"{why} c{circ:.2f} a{area / 1000:.0f}k", (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    title = f"mask: block {config.ADAPTIVE_BLOCK_SIZE}, C {config.ADAPTIVE_C}"
    if factor > 1:   # enlarge for display next to the full frame
        vis = cv2.resize(vis, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
        title += f", input 1/{factor}"
    cv2.putText(vis, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return vis


def draw(frame_gray, res, idx):
    """Return a colour image with the detection drawn on it."""
    img = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    colour = GREEN if res["method"] == "traditional" else RED
    if res["found"] and res["ellipse"] is not None:
        cv2.ellipse(img, res["ellipse"], colour, 1, cv2.LINE_AA)          # pupil boundary
        x, y = res["center"]
        center = (int(round(x)), int(round(y)))                           # pixel coordinates must be ints
        cv2.drawMarker(img, center, colour, cv2.MARKER_CROSS, 10, 1)      # pupil center
        label = f"#{idx} {res['method']} conf {res['confidence']:.2f} d {res['diameter']:.0f}px"
    else:
        label = f"#{idx} NOT FOUND (conf {res['confidence']:.2f})"
        colour = YELLOW
    # Draw the text twice: thick black outline first, then the colour on top (readable on any background)
    cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Draw CGF detections on a video for visual checking")
    ap.add_argument("video", type=Path)
    ap.add_argument("--sheet", type=int, default=24, help="number of frames in the spot-check grid")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--pupil-diameter", type=float, default=None,
                    help="adjust size limits / expected size (see evaluate_own.py)")
    ap.add_argument("--only", choices=["area", "prior"], default="area",
                    help="with --pupil-diameter: 'area' (default) or 'prior' (see evaluate_own.py)")
    ap.add_argument("--downscale", type=int, default=1, help="shrink input before detection (2 worked best)")
    ap.add_argument("--show-mask", action="store_true",
                    help="show the traditional branch's binary mask next to each frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: next to the video")
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")
    if args.pupil_diameter:
        scale_parameters(args.pupil_diameter, args.only)
    detector = CGFDetector(weights_path=str(CGF_DIR / "best_model_finetune_v2_ep122_miou9524.pkl"),
                           threshold=args.threshold)

    out_dir = args.out_dir or args.video.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # File names describe the run, so runs don't overwrite each other
    stem = args.video.stem
    if args.downscale > 1:
        stem += f"_ds{args.downscale}"
    if args.pupil_diameter:
        stem += f"_d{args.pupil_diameter:g}_{args.only}"
    if args.show_mask:
        stem += "_mask"

    cap = cv2.VideoCapture(str(args.video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        n = min(n, args.max_frames)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    # Frame numbers for the spot-check grid: --sheet frames, evenly spread over the video
    sheet_idx = set(np.linspace(0, max(n - 1, 0), min(args.sheet, n)).astype(int).tolist())

    writer = None
    sheet_imgs = []
    counts = {"traditional": 0, "RITNet": 0, "not_found": 0}
    for i in tqdm(range(n), desc=args.video.name):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        res = detect_downscaled(detector, gray, args.downscale)

        outcome = res["method"] if res["found"] else "not_found"
        counts[outcome] += 1

        img = draw(gray, res, i)
        if args.show_mask:
            img = np.hstack([img, mask_view(gray, detector, args.downscale)])        # frame | mask side by side
        if writer is None:                       # create on first frame, so size is the real one
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / f"{stem}_overlay.avi"),
                                     cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        writer.write(img)
        if i in sheet_idx:
            sheet_imgs.append(img)
    cap.release()
    if writer:
        writer.release()

    # Spot-check grid: 6 columns (3 with mask, since each tile is twice as wide),
    # frames shrunk to half size
    if sheet_imgs:
        cols = 3 if args.show_mask else 6
        h, w = sheet_imgs[0].shape[:2]
        tiles = [cv2.resize(t, (w // 2, h // 2)) for t in sheet_imgs]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))   # pad with black tiles so the last row is full
        rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]   # one image per grid row
        cv2.imwrite(str(out_dir / f"{stem}_sheet.png"), np.vstack(rows))

    total = sum(counts.values())
    print(f"\n{total} frames: traditional {counts['traditional']}, RITNet {counts['RITNet']}, "
          f"not found {counts['not_found']}")
    print(f"Saved {out_dir / (stem + '_overlay.avi')}")
    print(f"Saved {out_dir / (stem + '_sheet.png')}")


if __name__ == "__main__":
    main()
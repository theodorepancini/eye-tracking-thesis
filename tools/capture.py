"""
Capture video from the eye cameras with a timestamp per frame.

Purpose: measure what the cameras ACTUALLY deliver (fps, resolution, dropped
and duplicated frames) and record raw data, without OBS in between.
This is a measurement tool, not the final acquisition module.

For each camera it saves (in data/own/<date_time>/):
  cam<N>.avi            the video
  cam<N>_frames.csv     one row per frame: host timestamp, time since previous
                        frame, and how much the image changed (0 = duplicate)
  cam<N>_summary.json   requested vs actual settings, measured fps, drops, duplicates

Usage (from the eye-tracking-thesis folder, venv active):
  python tools/capture.py --list                                        # which camera numbers exist
  python tools/capture.py --devices 0 1 --seconds 30 --codec FFV1       # both eyes, lossless (use this for data)
  python tools/capture.py --devices 0 1 --seconds 30 --preview          # show the image while recording (q = stop)
  python tools/capture.py --devices 0 1 --seconds 1800 --no-video       # 30 min stability test, no files

Setup findings (2026-11-06):
  - The capture devices MUST be on SS (USB 3) ports: ~6.5 fps otherwise, 60 fps on SS.
  - The device always delivers 1920x1080 YUY2 and ignores requests for other
    sizes/formats (no MJPEG), so those options were removed.
  - The 640x480 camera image is pillarboxed in that frame; it is cropped and
    scaled back to 640x480 before saving (correct proportions).
  - Close OBS first: a camera can usually only be opened by one program.
  - --codec FFV1 is lossless and keeps up at 640x480; MJPG is lossy.
"""
import argparse
import csv
import json
import platform
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]        # .../eye-tracking-thesis

def pick_backend():
    """DirectShow on Windows, V4L2 on Linux.

    (MSMF on Windows was tested and hands over the same frame several times.)
    """
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2


def open_source(dev, backend):
    """A number opens a camera; anything else is treated as a video file (for testing)."""
    if str(dev).isdigit():
        return cv2.VideoCapture(int(dev), backend)
    return cv2.VideoCapture(str(dev))


def list_cameras(backend, max_index=8):
    print("Probing camera indices...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            shape = f"{frame.shape[1]}x{frame.shape[0]}" if ok else "no frame"
            print(f"  device {i}: opened, default {shape}, reported fps {cap.get(cv2.CAP_PROP_FPS):.1f}")
        cap.release()


class Camera:
    """Reads frames in its own thread; a second thread writes them to disk.

    Separating read and write means slow disk/encoding does not delay reading,
    so the timestamps reflect the camera, not our writer.
    """

    def __init__(self, dev, args, out_dir, stop_event):
        self.dev = dev
        self.args = args
        self.stop = stop_event
        self.name = f"cam{dev}" if str(dev).isdigit() else f"cam_{Path(str(dev)).stem}"
        self.out_dir = out_dir
        self.cap = open_source(dev, pick_backend())
        if not self.cap.isOpened():
            sys.exit(f"Could not open device {dev}. Is OBS or another program using it?")

        # The size the device delivers (always 1920x1080 for our capture devices).
        # The driver's fps/format values were unreliable (0.0 / '?'), so they're not used.
        self.reported = {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        self.queue = queue.Queue(maxsize=600)   # ~10 s of frames at 60 fps
        self.rows = []                          # per-frame log
        self.queue_overflows = 0
        self.saved = 0                          # frames handed to the writer
        self.writer = None
        self.latest = None                      # for preview

        # "640x480" -> (640, 480); "full" -> None (keep the size the camera delivers)
        self.save_size = None
        if args.save_size != "full":
            self.save_size = tuple(int(v) for v in args.save_size.split("x"))

    def capture_loop(self):
        prev_t = None
        prev_gray = None
        i = 0
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            t = time.perf_counter()             # timestamp right after the frame arrived
            if not ok:
                if not str(self.dev).isdigit():  # end of test file
                    break
                continue

            # Compare with the previous frame at FULL resolution (downscaling would
            # average away the sensor noise). Real sensor noise means two real
            # frames are never pixel-identical, so diff == 0 is a duplicated frame.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            diff = float("nan")                 # first frame has nothing to compare with
            if prev_gray is not None:
                diff = float(cv2.mean(cv2.absdiff(gray, prev_gray))[0])
            dt_ms = (t - prev_t) * 1000 if prev_t else float("nan")

            if self.args.no_video:
                video_frame = -1                 # timing test only, nothing saved
            else:
                try:
                    self.queue.put_nowait(frame)
                    video_frame = self.saved     # index of this frame in the video file
                    self.saved += 1
                except queue.Full:
                    self.queue_overflows += 1    # writer too slow -> frame not saved
                    video_frame = -1             # -1 = not in the video file

            self.rows.append({"frame": i, "video_frame": video_frame, "t_s": t,
                              "dt_ms": dt_ms, "mean_abs_diff": diff})
            self.latest = frame
            prev_t = t
            prev_gray = gray
            i += 1
        self.queue.put(None)                     # tell the writer we're done

    def write_loop(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            # Scale to the camera's native size before saving. The 640x480 (4:3)
            # camera image arrives PILLARBOXED in 1920x1080: a centred 1440x1080
            # picture with black bars left and right. Crop the centre to the
            # target aspect ratio first, then scale - otherwise the image gets
            # squeezed horizontally (round pupils become tall ellipses).
            if self.save_size is not None:
                h, w = frame.shape[:2]
                tw, th = self.save_size
                crop_w = int(round(h * tw / th))         # width of the 4:3 picture
                if crop_w < w:
                    x0 = (w - crop_w) // 2
                    frame = frame[:, x0:x0 + crop_w]
                frame = cv2.resize(frame, self.save_size, interpolation=cv2.INTER_AREA)
            if self.writer is None:              # create on first frame, so size is the real one
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*self.args.codec)
                fps = 60                         # camera rate (measured 60.0 unique fps)
                self.writer = cv2.VideoWriter(str(self.out_dir / f"{self.name}.avi"), fourcc, fps, (w, h))
            self.writer.write(frame)
        if self.writer:
            self.writer.release()

    def summary(self):
        nan = float("nan")
        n = len(self.rows)
        # Skip the first frame: it has no previous frame, so no interval or diff
        dts = np.array([r["dt_ms"] for r in self.rows[1:]])
        diffs = np.array([r["mean_abs_diff"] for r in self.rows[1:]])
        duration = self.rows[-1]["t_s"] - self.rows[0]["t_s"] if n > 1 else 0.0

        # Defaults for when there are fewer than 2 frames
        med = p95 = dt_max = diff_median = measured_fps = unique_fps = nan
        dropped = duplicates = 0
        if dts.size:
            med = float(np.median(dts))
            p95 = float(np.percentile(dts, 95))
            dt_max = float(dts.max())
            # Frames probably lost: an interval of ~2x median = 1 missing frame, ~3x = 2, etc.
            # Only intervals >= 1.9x median count: 1.5-1.8x (e.g. 26-30 ms at 60 fps)
            # was seen to be timing jitter, the frame count showed nothing was lost.
            ratios = dts / med
            dropped = int(np.sum(np.where(ratios >= 1.9, np.round(ratios) - 1, 0)))
            # Pixel-identical consecutive frames = certain duplicates (device repeated a frame)
            duplicates = int(np.sum(diffs == 0))
            # Typical frame-to-frame change; compare with duplicates to judge the noise level
            diff_median = float(np.median(diffs))
        if duration > 0:
            measured_fps = (n - 1) / duration
            unique_fps = (int(np.sum(diffs > 0)) + 1) / duration   # +1 = the first frame

        return {
            "device": str(self.dev),
            "delivered_size": self.reported,
            "frames": n,
            "duration_s": duration,
            "measured_fps": measured_fps,
            "dt_ms": {"median": med, "p95": p95, "max": dt_max},
            "estimated_dropped": dropped,
            "save_size": self.args.save_size,
            "duplicates_exact": duplicates,
            "diff_median": diff_median,
            "unique_fps": unique_fps,
            "frames_not_saved_queue_full": self.queue_overflows,
            "codec": self.args.codec,
            "opencv": cv2.__version__, "platform": platform.platform(),
        }

    def save_logs(self):
        with open(self.out_dir / f"{self.name}_frames.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "video_frame", "t_s", "dt_ms", "mean_abs_diff"])
            w.writeheader()
            w.writerows(self.rows)
        s = self.summary()
        (self.out_dir / f"{self.name}_summary.json").write_text(json.dumps(s, indent=2))
        return s


def main():
    ap = argparse.ArgumentParser(description="Capture eye-camera video with per-frame timestamps")
    ap.add_argument("--list", action="store_true", help="list camera indices and exit")
    ap.add_argument("--devices", nargs="+", default=["0"], help="camera numbers (or a video file for testing)")
    ap.add_argument("--seconds", type=float, default=10, help="recording length")
    ap.add_argument("--no-video", action="store_true",
                    help="don't save video, only timestamps/summary (long stability tests)")
    ap.add_argument("--save-size", default="640x480",
                    help="size of the saved video, WxH (default 640x480 = camera native, 4:3), or 'full'")
    ap.add_argument("--codec", choices=["MJPG", "FFV1"], default="MJPG",
                    help="file codec: MJPG (lossy, light), FFV1 (lossless, use for data)")
    ap.add_argument("--preview", action="store_true", help="show the image (q to stop early)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: data/own/<date_time>")
    args = ap.parse_args()

    if args.list:
        list_cameras(pick_backend())
        return

    out_dir = args.out_dir or REPO_DIR / "data" / "own" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    cams = [Camera(d, args, out_dir, stop) for d in args.devices]
    for c in cams:
        print(f"{c.name}: device delivers {c.reported['width']}x{c.reported['height']}")

    # Two threads per camera: one reads frames, one writes them to disk
    capture_threads = [threading.Thread(target=c.capture_loop) for c in cams]
    write_threads = [threading.Thread(target=c.write_loop) for c in cams]
    threads = capture_threads + write_threads
    for t in threads:
        t.start()

    print(f"Recording {args.seconds:g} s... (Ctrl+C to stop early)")
    t_end = time.perf_counter() + args.seconds
    try:
        # Keep going until time is up, or every camera has stopped (e.g. test file ended)
        while time.perf_counter() < t_end and any(t.is_alive() for t in capture_threads):
            if args.preview:
                for c in cams:
                    if c.latest is not None:
                        cv2.imshow(c.name, c.latest)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    stop.set()
    if not args.no_video:
        print("Recording stopped. Saving remaining frames to disk - don't press Ctrl+C...")
    for t in threads:
        t.join()
    for c in cams:
        c.cap.release()
    cv2.destroyAllWindows()

    print()
    for c in cams:
        s = c.save_logs()
        print(f"{c.name}: {s['frames']} frames in {s['duration_s']:.1f} s -> measured {s['measured_fps']:.1f} fps")
        print(f"  frame interval median {s['dt_ms']['median']:.1f} ms, p95 {s['dt_ms']['p95']:.1f} ms, "
              f"max {s['dt_ms']['max']:.1f} ms")
        print(f"  estimated dropped frames: {s['estimated_dropped']}, "
              f"exact duplicates: {s['duplicates_exact']}, not saved: {s['frames_not_saved_queue_full']}")
        print(f"  unique frames per second: {s['unique_fps']:.1f}, "
              f"median frame-to-frame change: {s['diff_median']:.3f}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
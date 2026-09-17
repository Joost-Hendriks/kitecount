#!/usr/bin/env python3
"""
kitecount.py - real-time kitesurfer counter for a YouTube live stream (or any video source).
 
Pipeline
  yt-dlp  -> resolves the live HLS URL (re-resolved automatically when it expires)
  OpenCV  -> reader thread that always holds only the *latest* frame (no lag build-up)
  YOLO    -> detects COCO 'kite' (+ optionally 'person' / 'surfboard'), ByteTrack IDs
  Output  -> overlay window, smoothed count, CSV log, optional JSON over HTTP
 
A kitesurfer is counted via their kite: kites are large, high-contrast and far easier
to detect from a beach cam than a 20-pixel rider. Use --roi to exclude kites parked on
the beach, or switch --mode riders to count persons inside a water ROI instead.

By default the window only pops up/updates when a kite is currently detected (idle
otherwise, showing the last detection) rather than rendering a continuous live feed -
this avoids CPU-bound inference or stream hiccups ever making the window look frozen.
Pass --show-always for the old continuous-video display, or --no-show for no window.
 
Usage
  python kitecount.py https://www.youtube.com/watch?v=mnxV4iit7S4
  python kitecount.py URL --roi roi.json --csv counts.csv --http 8080
  python kitecount.py URL --pick-roi              # click a polygon, saved to roi.json
  python kitecount.py some_clip.mp4 --no-show     # offline test
"""
from __future__ import annotations
 
import argparse
import csv
import json
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
 
import cv2
import numpy as np
 
COCO = {"person": 0, "kite": 33, "surfboard": 37}
 
 
# --------------------------------------------------------------------------- source
def resolve_stream(url: str, max_height: int) -> str:
    """Return a direct media URL. Non-YouTube inputs (files, rtsp, http .m3u8) pass through."""
    if not any(h in url for h in ("youtube.com", "youtu.be")):
        return url
    import yt_dlp
 
    opts = {
        "quiet": True,
        "no_warnings": True,
        # OpenCV only needs the video track, and live streams are usually served as
        # separate video-only/audio-only HLS renditions (no muxed format available),
        # so prefer bestvideo first and only fall back to a muxed 'best' if present.
        "format": f"bestvideo[height<={max_height}][protocol^=m3u8]/"
                  f"best[height<={max_height}][protocol^=m3u8]/"
                  f"bestvideo[height<={max_height}]/best[height<={max_height}]/best",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    print(f"[src] {info.get('title')!r}  live={info.get('is_live')}  "
          f"{info.get('width')}x{info.get('height')}")
    return info["url"]
 
 
class LatestFrameReader(threading.Thread):
    """Continuously grabs frames; keeps only the newest. Reconnects on failure.

    cap.read() on a live HLS source can hang indefinitely (a stalled segment /
    dead connection) instead of returning False, which would otherwise wedge
    this thread forever on one stale frame. A watchdog thread tracks time
    since the last frame and force-releases the VideoCapture from outside if
    it stalls too long; releasing a capture while another thread is blocked
    inside read() reliably makes that call return, so the normal reconnect
    path below takes over automatically.
    """

    def __init__(self, src: str, max_height: int, stall_timeout: float = 12.0):
        super().__init__(daemon=True)
        self.src, self.max_height = src, max_height
        self.stall_timeout = stall_timeout
        self.lock = threading.Lock()
        self.frame, self.frame_id = None, 0
        self.running = True
        self.is_file = Path(src).exists()
        self.eof = False
        self.cap = None
        self.force_reconnect = False
        self.last_frame_time = time.time()
        if not self.is_file:
            threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        while self.running:
            time.sleep(1.0)
            if self.cap is not None and time.time() - self.last_frame_time > self.stall_timeout:
                print(f"[src] no frame for {self.stall_timeout:.0f}s, forcing reconnect")
                self.force_reconnect = True
                try:
                    self.cap.release()
                except Exception:
                    pass

    def _open(self):
        media = resolve_stream(self.src, self.max_height)
        cap = cv2.VideoCapture(media, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        backoff = 2
        while self.running:
            try:
                cap = self._open()
            except Exception as e:  # yt-dlp / network error
                print(f"[src] resolve failed: {e}; retry in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            self.cap = cap
            self.force_reconnect = False
            self.last_frame_time = time.time()
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            fails = 0
            while self.running and not self.force_reconnect:
                ok, f = cap.read()
                if not ok:
                    fails += 1
                    if self.is_file:
                        self.eof = True
                        self.running = False
                        break
                    if fails > 50:  # HLS URL expired or stream hiccup -> re-resolve
                        print("[src] stream lost, reconnecting")
                        break
                    time.sleep(0.05)
                    continue
                fails, backoff = 0, 2
                self.last_frame_time = time.time()
                with self.lock:
                    self.frame, self.frame_id = f, self.frame_id + 1
                if self.is_file:  # play files at real speed so the counter behaves like live
                    time.sleep(1.0 / fps)
            self.cap = None
            cap.release()
 
    def latest(self):
        with self.lock:
            return self.frame_id, None if self.frame is None else self.frame.copy()
 
 
# --------------------------------------------------------------------------- ROI
def load_roi(path: str | None):
    if not path:
        return None
    pts = json.loads(Path(path).read_text())
    return np.array(pts, dtype=np.float32)  # normalised 0..1 coordinates
 
 
def pick_roi(frame, out_path: str):
    pts, h, w = [], *frame.shape[:2]
    win = "click ROI polygon - ENTER=save, BACKSPACE=undo, ESC=cancel"
 
    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
 
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, cb)
    while True:
        vis = frame.copy()
        if pts:
            cv2.polylines(vis, [np.array(pts)], len(pts) > 2, (0, 255, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and len(pts) > 2:
            norm = [[x / w, y / h] for x, y in pts]
            Path(out_path).write_text(json.dumps(norm))
            print(f"[roi] saved {len(pts)} points to {out_path}")
            break
        if k == 8 and pts:
            pts.pop()
        if k == 27:
            break
    cv2.destroyWindow(win)
 
 
def in_roi(roi_px, x, y):
    return roi_px is None or cv2.pointPolygonTest(roi_px, (float(x), float(y)), False) >= 0
 
 
# --------------------------------------------------------------------------- stats
class Stats:
    def __init__(self, window_s: float, track_ttl_s: float, history_len: int = 200_000):
        self.samples = deque()  # (t, count) - short smoothing window
        self.window_s = window_s
        self.track_ttl_s = track_ttl_s
        self.last_seen: dict[int, float] = {}  # track id -> last time seen
        self.unique_total = 0
        self.lock = threading.Lock()
        self.snapshot = {}
        # (t, instant, smoothed) for the whole run - used for --plot / offline analysis.
        # 200k samples is ~2.3 days at 1/s, plenty for an unattended monitoring run.
        self.history: deque = deque(maxlen=history_len)

    def update(self, t, count, track_ids):
        with self.lock:
            self.samples.append((t, count))
            while self.samples and t - self.samples[0][0] > self.window_s:
                self.samples.popleft()
            for tid in track_ids:
                if tid not in self.last_seen:
                    self.unique_total += 1
                self.last_seen[tid] = t
            for tid in [k for k, v in self.last_seen.items() if t - v > self.track_ttl_s]:
                del self.last_seen[tid]
            vals = [c for _, c in self.samples]
            smoothed = int(round(statistics.median(vals)))
            self.history.append((t, count, smoothed))
            self.snapshot = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "instant": count,
                "smoothed": smoothed,
                "max_window": max(vals),
                "active_tracks": len(self.last_seen),
                "unique_since_start": self.unique_total,
            }
            return dict(self.snapshot)

    def history_copy(self):
        with self.lock:
            return list(self.history)
 
 
def start_http(stats: Stats, port: int):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            with stats.lock:
                body = json.dumps(stats.snapshot).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
 
        def log_message(self, *a):
            pass
 
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[http] JSON count at http://localhost:{port}/")


# --------------------------------------------------------------------------- plot
def render_plot(history, path: str, title: str):
    """Render a PNG line chart of kite count over the run so far. No GUI window -
    this uses matplotlib's Agg (file-only) backend and is safe to call repeatedly
    from the main loop; the figure is closed each time to avoid leaking memory."""
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    LIGHT_SURFACE = "#fcfcfb"
    INK_PRIMARY = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#898781"
    GRIDLINE = "#e1e0d9"
    AXIS = "#c3c2b7"
    SERIES_SMOOTHED = "#2a78d6"  # categorical slot 1 (blue) - the headline number

    times = [datetime.fromtimestamp(t) for t, _, _ in history]
    instant = [i for _, i, _ in history]
    smoothed = [s for _, _, s in history]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    fig.patch.set_facecolor(LIGHT_SURFACE)
    ax.set_facecolor(LIGHT_SURFACE)

    # Instant count: thin, muted - context for the smoothed line, not a competing series.
    ax.plot(times, instant, color=INK_MUTED, linewidth=1, alpha=0.5, label="instant")
    # Smoothed count: bold, the primary series - matches the on-screen headline number.
    ax.plot(times, smoothed, color=SERIES_SMOOTHED, linewidth=2, label="smoothed")

    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_ylabel("kitesurfers", color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    for name, spine in ax.spines.items():
        if name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(AXIS)

    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="YouTube URL, video file, rtsp:// or .m3u8")
    ap.add_argument("--model", default="yolo11m.pt", help="YOLO weights (yolo11s/m/l/x.pt or your fine-tuned .pt)")
    ap.add_argument("--mode", choices=["kites", "riders"], default="kites",
                    help="kites: count COCO 'kite'; riders: count 'person' inside ROI")
    ap.add_argument("--classes", default=None,
                    help="override detected class ids, e.g. '0' for a fine-tuned 1-class model")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size; kites are small, keep this high")
    ap.add_argument("--max-height", type=int, default=1080, help="stream resolution to request")
    ap.add_argument("--stall-timeout", type=float, default=12.0,
                    help="seconds without a new frame before forcing a stream reconnect")
    ap.add_argument("--min-area", type=float, default=0.0, help="ignore boxes smaller than this (px^2)")
    ap.add_argument("--roi", help="JSON polygon (normalised coords) - only count detections inside")
    ap.add_argument("--pick-roi", action="store_true", help="draw ROI on first frame and save to roi.json")
    ap.add_argument("--interval", type=float, default=0.0, help="min seconds between inferences (0 = as fast as possible)")
    ap.add_argument("--window", type=float, default=10.0, help="smoothing window in seconds")
    ap.add_argument("--csv", help="append a row per inference to this CSV")
    ap.add_argument("--http", type=int, help="serve the current count as JSON on this port")
    ap.add_argument("--save-frames", help="dir: save a frame every --save-every s (training data)")
    ap.add_argument("--save-every", type=float, default=30.0)
    ap.add_argument("--no-show", action="store_true", help="headless: no window at all")
    ap.add_argument("--show-always", action="store_true",
                    help="show every processed frame like a live feed, instead of the default "
                         "(only pop up/update the window when a kite is currently detected)")
    ap.add_argument("--device", default=None, help="'0' for CUDA GPU, 'cpu', 'mps'")
    ap.add_argument("--plot", help="PNG path: periodically render a count-over-time line chart here "
                                    "(implies --no-show; auto-enables --csv next to it unless set)")
    ap.add_argument("--plot-every", type=float, default=15.0, help="seconds between chart re-renders")
    args = ap.parse_args()

    if args.plot:
        args.no_show = True
        if not args.csv:
            args.csv = str(Path(args.plot).with_suffix(".csv"))
            print(f"[plot] no --csv given, logging to {args.csv}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.classes is not None:
        classes = [int(c) for c in args.classes.split(",")]
        count_cls = set(classes)
    elif args.mode == "kites":
        classes = [COCO["kite"], COCO["person"], COCO["surfboard"]]
        count_cls = {COCO["kite"]}
    else:
        classes = [COCO["kite"], COCO["person"], COCO["surfboard"]]
        count_cls = {COCO["person"]}
 
    reader = LatestFrameReader(args.source, args.max_height, stall_timeout=args.stall_timeout)
    reader.start()
    print("[src] waiting for first frame ...")
    while reader.latest()[1] is None:
        if reader.eof:
            raise SystemExit("could not read source")
        time.sleep(0.1)
 
    if args.pick_roi:
        pick_roi(reader.latest()[1], "roi.json")
        args.roi = args.roi or "roi.json"
    roi_norm = load_roi(args.roi)
 
    stats = Stats(args.window, track_ttl_s=5.0)
    if args.http:
        start_http(stats, args.http)
 
    csv_f = None
    if args.csv:
        new = not Path(args.csv).exists()
        csv_f = open(args.csv, "a", newline="")
        w = csv.writer(csv_f)
        if new:
            w.writerow(["time", "instant", "smoothed", "max_window", "active_tracks", "unique_since_start"])
 
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    last_save = 0.0
    last_id, last_inf, fps_ema = -1, 0.0, None
    last_print = 0.0
    last_plot = 0.0
    plot_title = f"Kitesurfers over time - {Path(args.source).name if Path(args.source).exists() else args.source}"
 
    try:
        while reader.running or not reader.eof:
            fid, frame = reader.latest()
            now = time.time()
            if frame is None or fid == last_id or now - last_inf < args.interval:
                if reader.eof:
                    break
                if not args.no_show:
                    # Pump the HighGUI window's message queue even while waiting for
                    # the next frame/inference slot. Without this, Windows sees a
                    # window that never processes messages during a slow inference
                    # pass or an HLS segment-boundary pause and marks it "Not
                    # Responding" (frozen-looking) until the loop happens to reach
                    # the imshow()/waitKey() call further down.
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                time.sleep(0.005)
                continue
            last_id, last_inf = fid, now
            h, w_ = frame.shape[:2]
            roi_px = None if roi_norm is None else (roi_norm * [w_, h]).astype(np.int32)
 
            if args.save_frames and now - last_save > args.save_every:
                cv2.imwrite(str(Path(args.save_frames) / f"{datetime.now():%Y%m%d_%H%M%S}.jpg"), frame)
                last_save = now
 
            t0 = time.time()
            res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=classes,
                              conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            dt = time.time() - t0
            fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt
 
            count, ids = 0, []
            boxes = res.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                tid = int(boxes.id[i]) if boxes.id is not None else None
                counted = cls in count_cls and area >= args.min_area and in_roi(roi_px, cx, cy)
                if counted:
                    count += 1
                    if tid is not None:
                        ids.append(tid)
                if not args.no_show:
                    col = (0, 255, 0) if counted else (128, 128, 128)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                    label = f"{res.names[cls]}{'' if tid is None else f' #{tid}'} {float(boxes.conf[i]):.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
 
            snap = stats.update(now, count, ids)
            if csv_f:
                csv.writer(csv_f).writerow(list(snap.values()))
                csv_f.flush()
            if now - last_print > 1:
                print(f"\r{snap['time']}  now={snap['instant']:3d}  smoothed={snap['smoothed']:3d}  "
                      f"max{int(args.window)}s={snap['max_window']:3d}  unique={snap['unique_since_start']:4d}  "
                      f"{fps_ema:4.1f} fps", end="", flush=True)
                last_print = now

            if args.plot and now - last_plot > args.plot_every:
                render_plot(stats.history_copy(), args.plot, plot_title)
                last_plot = now

            if not args.no_show:
                if args.show_always or count > 0:
                    if roi_px is not None:
                        cv2.polylines(frame, [roi_px], True, (0, 255, 255), 1)
                    txt = f"Kitesurfers: {snap['smoothed']}  (now {count})"
                    cv2.rectangle(frame, (0, 0), (430, 40), (0, 0, 0), -1)
                    cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow("kitecount", frame)
                # Pump the window's message queue every processed frame, whether or
                # not a detection happened, so it never looks frozen and stays
                # quittable even while the last-shown detection frame sits idle.
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        reader.running = False
        if csv_f:
            csv_f.close()
        if not args.no_show:
            cv2.destroyAllWindows()
        if args.plot:
            render_plot(stats.history_copy(), args.plot, plot_title)
            print(f"[plot] saved {args.plot}")
        print()
 
 
if __name__ == "__main__":
    main()
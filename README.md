# kitecount
Counts and logs kites on a live stream
# kitecount — is anyone else out there?

If you kite alone, going out when nobody else is on the water means nobody's around to
help if something goes wrong — a snapped line, a hard crash, a dragging session. This
watches a public beach-cam livestream and counts kites, so before you rig up you can
check whether other kiters are already out (or have been recently) instead of guessing
from a static webcam thumbnail. Point it at your local spot's cam, leave it running, and
check the count — or the chart — before you head down to the beach.

```bash
pip install -r requirements.txt        # install a CUDA build of torch first if you have an NVIDIA GPU
python kitecount.py https://www.youtube.com/watch?v=mnxV4iit7S4 --plot kitecounts.png
```
 
`--pick-roi`: click a polygon around the water/sky area and press ENTER. It's saved to `roi.json`, so later runs can use `--roi roi.json`. This keeps kites lying on the beach out of the count.

## Display modes
- **default:** the window only pops up/updates when a kite is currently detected (idle otherwise, showing the last detection) instead of rendering a continuous live feed. This is deliberate: on a CPU-only box, YOLO inference and HLS segment fetches both introduce multi-second pauses, and a window that stops pumping messages during those pauses gets marked "Not Responding" by Windows even though the app is fine — showing frames only on a detection sidesteps that entirely.
- `--show-always`: go back to a continuous live-video window (every processed frame, not just detections).
- `--no-show`: fully headless — no window at all, just console/CSV/HTTP output.
- `--plot chart.png`: no window either way — instead, periodically (re)render a PNG line chart of kite count over time (see below). Implies `--no-show`.

## How it counts
- **kites mode (default):** counts the COCO `kite` class. From a beach cam the kite is much easier to detect than the rider, and there's one kite per kitesurfer.
- **riders mode:** counts `person` detections inside the ROI (for close-range cams).
- Detections are tracked with ByteTrack. Output shows:
  - `now`: count in the current frame
  - `smoothed`: median over `--window` seconds (the number to use)
  - `max`: peak count in the window
  - `unique`: number of track IDs seen since start (an upper bound, because IDs switch after occlusions)

## Logging & charting
```bash
python kitecount.py URL --plot kitecounts.png
```
- No window — headless counting only.
- Logs every inference to CSV automatically (defaults to `kitecounts.csv` next to the PNG, unless you pass `--csv` yourself).
- Every `--plot-every` seconds (default 15), re-renders `kitecounts.png`: kite count over the whole run, smoothed line bold, instant line as light context, no GUI needed to check on it — just open the PNG. It also re-renders once on exit (including Ctrl+C).
- History for the chart is kept in memory (up to ~2.3 days at 1 sample/s), independent of the short `--window` used for the live `smoothed` figure.

## Useful flags
| flag | purpose |
|---|---|
| `--model yolo11s.pt / yolo11l.pt` | speed vs. accuracy (default m) |
| `--device 0` | CUDA GPU |
| `--imgsz 1280` | keep this high, since kites far away are small |
| `--interval 1` | infer once per second (CPU-friendly) |
| `--csv counts.csv` | log every inference |
| `--http 8080` | `GET http://localhost:8080/` returns the current counts as JSON |
| `--no-show` | headless, no window |
| `--show-always` | continuous live-video window instead of show-on-detect |
| `--plot chart.png [--plot-every 15]` | headless + periodic count-over-time PNG chart |
| `--stall-timeout 12` | seconds with no new frame before forcing a stream reconnect (handles a wedged/blocked HLS read) |
| `--save-frames data/ --save-every 30` | collect frames to build a training set |

**CPU-only note:** without a CUDA GPU (`--device cpu`, or torch has no CUDA build), `yolo11m.pt` at `imgsz=1280` runs at roughly **1 inference/sec** — real-time detection, not real-time video. Drop to `--model yolo11s.pt` and/or a lower `--imgsz` for more throughput at some accuracy cost.

## Improving accuracy
COCO's `kite` class is trained mostly on hobby kites. It usually picks up LEI kites, but it can miss distant or edge-on ones. If that happens:
1. Collect frames with `--save-frames`.
2. Label ~200–500 kites in CVAT or Roboflow.
3. Fine-tune: `yolo train model=yolo11m.pt data=kites.yaml imgsz=1280 epochs=100`.
4. Run with `--model runs/detect/train/weights/best.pt --classes 0`.
If far-away kites are still missed, crop/tile the frame (e.g. SAHI) before raising `imgsz` further.
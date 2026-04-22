import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


WINDOW_NAME = "Graphical Abstract Frame Picker"
DISPLAY_MAX_W = 1800
DISPLAY_MAX_H = 1100


def fit_to_screen(frame):
    h, w = frame.shape[:2]
    scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_hud(frame, frame_idx, total_frames, mode, saved_count):
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (900, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    lines = [
        f"frame {frame_idx + 1}/{total_frames}",
        f"view: {mode}",
        f"saved pairs: {saved_count}",
        "A/D prev/next | J/L -30/+30 | Z/C -180/+180 | R toggle raw/det | S save pair | G jump | Esc quit",
    ]
    y = 55
    for line in lines:
        cv2.putText(frame, line, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return frame


def render_detection(frame, model, imgsz, conf):
    result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    return result.plot(conf=True, line_width=4, font_size=16)


def prompt_jump(current_idx, total_frames):
    try:
        raw = input(f"Jump to frame [1..{total_frames}] (current {current_idx + 1}): ").strip()
        if not raw:
            return current_idx
        target = int(raw) - 1
        return min(max(target, 0), total_frames - 1)
    except Exception:
        return current_idx


def main():
    parser = argparse.ArgumentParser(description="Interactively pick raw+detection frame pairs for graphical abstracts.")
    parser.add_argument("--video", required=True, help="Path to source video.")
    parser.add_argument("--model", required=True, help="Path to YOLO detector weights.")
    parser.add_argument("--output-dir", required=True, help="Where to save raw/detection pairs.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO confidence threshold.")
    parser.add_argument("--prefix", default="graphical_abstract", help="Filename prefix for saved pairs.")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    mode = "detection"
    saved_count = len(list(output_dir.glob(f"{args.prefix}_*_raw.jpg")))
    cached_idx = None
    cached_raw = None
    cached_det = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        if cached_idx != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            cached_raw = frame
            cached_det = render_detection(frame, model, args.imgsz, args.conf)
            cached_idx = frame_idx

        shown = cached_raw.copy() if mode == "raw" else cached_det.copy()
        shown = draw_hud(shown, frame_idx, total_frames, mode, saved_count)
        shown = fit_to_screen(shown)
        cv2.imshow(WINDOW_NAME, shown)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        if key == ord("a"):
            frame_idx = max(0, frame_idx - 1)
        elif key == ord("d"):
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif key == ord("j"):
            frame_idx = max(0, frame_idx - 30)
        elif key == ord("l"):
            frame_idx = min(total_frames - 1, frame_idx + 30)
        elif key == ord("z"):
            frame_idx = max(0, frame_idx - 180)
        elif key == ord("c"):
            frame_idx = min(total_frames - 1, frame_idx + 180)
        elif key == ord("r"):
            mode = "raw" if mode == "detection" else "detection"
        elif key == ord("g"):
            frame_idx = prompt_jump(frame_idx, total_frames)
        elif key == ord("s"):
            save_idx = saved_count + 1
            raw_path = output_dir / f"{args.prefix}_{save_idx:03d}_raw.jpg"
            det_path = output_dir / f"{args.prefix}_{save_idx:03d}_detection.jpg"
            cv2.imwrite(str(raw_path), cached_raw)
            cv2.imwrite(str(det_path), cached_det)
            saved_count += 1
            print(f"Saved: {raw_path}")
            print(f"Saved: {det_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import argparse
import csv
import sys
from pathlib import Path

import cv2


def ensure_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def collect_frames(frames_dir: Path) -> list[Path]:
    return sorted(path for path in frames_dir.glob("*.jpg") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap draft tracking labels for article_pack clips.")
    parser.add_argument(
        "--model",
        default="runs/detect/article_pack/experiments/detector/cam10_round12_yolo11x/weights/best.pt",
        help="Detector weights",
    )
    parser.add_argument(
        "--tracking-root",
        default="article_pack/tracking_benchmark",
        help="Tracking benchmark root",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=["clip_A", "clip_B"],
        help="Clip name to process. Can be repeated.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack",
        help="Tracker backend or tracker YAML path",
    )
    parser.add_argument(
        "--tracker-tag",
        default=None,
        help="Output tag for draft files. Defaults to tracker name or YAML stem.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference size")
    args = parser.parse_args()

    ensure_project_root()
    from article_pack.scripts.tracker_pipeline import TrackerPipeline

    tracking_root = Path(args.tracking_root)
    pipe = TrackerPipeline(
        model_path=args.model,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    for clip_name in args.clip:
        clip_dir = tracking_root / clip_name
        frames_dir = clip_dir / "frames"
        tracker_tag = args.tracker_tag or Path(args.tracker).stem
        draft_path = clip_dir / f"draft_{tracker_tag}.txt"
        summary_path = clip_dir / f"draft_{tracker_tag}_summary.txt"

        frames = collect_frames(frames_dir)
        if not frames:
            raise RuntimeError(f"No frames found in {frames_dir}")

        pipe.reset()
        track_id_map: dict[int, int] = {}
        next_track_id = 1
        row_count = 0

        with draft_path.open("w", encoding="utf-8", newline="") as draft_f:
            draft_f.write("# frame_id,track_id,class_id,x1,y1,x2,y2\n")
            for frame_idx, frame_path in enumerate(frames, start=1):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"Cannot read frame: {frame_path}")
                detections = pipe.process_frame(frame)
                for det in detections:
                    if det.track_id < 0:
                        continue
                    if det.track_id not in track_id_map:
                        track_id_map[det.track_id] = next_track_id
                        next_track_id += 1
                    x1, y1, x2, y2 = det.bbox
                    draft_f.write(
                        f"{frame_idx},{track_id_map[det.track_id]},{det.cls},{x1},{y1},{x2},{y2}\n"
                    )
                    row_count += 1

        summary_path.write_text(
            "\n".join(
                [
                    f"clip={clip_name}",
                    f"tracker={args.tracker}",
                    f"tracker_tag={tracker_tag}",
                    f"model={args.model}",
                    f"frames={len(frames)}",
                    f"draft_rows={row_count}",
                    f"unique_draft_ids={len(track_id_map)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"{clip_name}: tracker={args.tracker} frames={len(frames)} "
            f"draft_rows={row_count} unique_ids={len(track_id_map)}"
        )


if __name__ == "__main__":
    main()

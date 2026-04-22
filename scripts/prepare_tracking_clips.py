import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2


@dataclass
class ClipSpec:
    clip_name: str
    start_sec: float
    num_frames: int


def parse_clip_spec(raw: str) -> ClipSpec:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid clip spec: {raw}")
    clip_name, start_sec, num_frames = parts
    return ClipSpec(clip_name=clip_name, start_sec=float(start_sec), num_frames=int(num_frames))


def wall_clock_for(source_name: str, offset_sec: float) -> str:
    stamp = source_name.split(".")[0].split("_")[-1]
    base_time = datetime.strptime(stamp, "%d%m%Y%H%M%S")
    return (base_time + timedelta(seconds=offset_sec)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fixed tracking clips for article_pack.")
    parser.add_argument(
        "--video",
        default="article_pack/data/cam10/source_video/1_10_R_23032026170000.mp4",
        help="Source video path",
    )
    parser.add_argument(
        "--output-root",
        default="article_pack/tracking_benchmark",
        help="Tracking benchmark root",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[
            "clip_A:600:180",
            "clip_B:2520:180",
        ],
        help="Clip spec as name:start_sec:num_frames. Can be repeated.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frames",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    specs = [parse_clip_spec(raw) for raw in args.clip]
    summary_rows = []

    for spec in specs:
        clip_dir = output_root / spec.clip_name
        frames_dir = clip_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = clip_dir / "frame_manifest.csv"
        gt_path = clip_dir / "gt_tracks.txt"

        if args.overwrite:
            for old_frame in frames_dir.glob("*.jpg"):
                old_frame.unlink()

        start_frame = int(round(spec.start_sec * fps))
        end_frame = min(start_frame + spec.num_frames, total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        with manifest_path.open("w", newline="", encoding="utf-8") as manifest_f:
            writer = csv.writer(manifest_f)
            writer.writerow(
                [
                    "frame_id",
                    "image_name",
                    "source_video",
                    "source_frame",
                    "video_offset_sec",
                    "wall_clock",
                    "fps",
                    "width",
                    "height",
                ]
            )

            frame_id = 1
            for source_frame in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                image_name = f"{spec.clip_name}_{frame_id:06d}.jpg"
                image_path = frames_dir / image_name
                cv2.imwrite(str(image_path), frame)
                offset_sec = source_frame / fps
                writer.writerow(
                    [
                        frame_id,
                        image_name,
                        video_path.name,
                        source_frame,
                        f"{offset_sec:.3f}",
                        wall_clock_for(video_path.name, offset_sec),
                        f"{fps:.3f}",
                        width,
                        height,
                    ]
                )
                frame_id += 1

        if not gt_path.exists() or gt_path.read_text(encoding="utf-8").strip() == "":
            gt_path.write_text("# frame_id,track_id,class_id,x1,y1,x2,y2\n", encoding="utf-8")

        summary_rows.append(
            {
                "clip_name": spec.clip_name,
                "start_sec": f"{spec.start_sec:.3f}",
                "start_wall_clock": wall_clock_for(video_path.name, spec.start_sec),
                "num_frames": end_frame - start_frame,
                "fps": f"{fps:.3f}",
                "duration_sec": f"{(end_frame - start_frame) / fps:.3f}",
            }
        )

    summary_path = output_root / "tracking_clips_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as summary_f:
        writer = csv.DictWriter(
            summary_f,
            fieldnames=["clip_name", "start_sec", "start_wall_clock", "num_frames", "fps", "duration_sec"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    cap.release()

    print(f"Prepared {len(summary_rows)} clips from {video_path.name}")
    for row in summary_rows:
        print(
            f"- {row['clip_name']}: start={row['start_wall_clock']} "
            f"frames={row['num_frames']} duration={row['duration_sec']}s"
        )


if __name__ == "__main__":
    main()

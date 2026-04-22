import argparse
import csv
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import cv2


TIMESTAMP_RE = re.compile(r"(\d{14})")


def parse_video_start(path: Path) -> datetime | None:
    match = TIMESTAMP_RE.search(path.stem)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%d%m%Y%H%M%S")


def even_timestamps(duration_sec: float, count: int) -> list[float]:
    if count <= 0 or duration_sec <= 0:
        return []
    step = duration_sec / count
    return [min((i + 0.5) * step, max(duration_sec - 0.001, 0)) for i in range(count)]


def extract_preview_frames(video_path: Path, output_dir: Path, count: int) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = total_frames / fps if fps else 0
    start_dt = parse_video_start(video_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for index, second in enumerate(even_timestamps(duration_sec, count), start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()
        if not ok:
            continue

        second_int = int(math.floor(second))
        hh = second_int // 3600
        mm = (second_int % 3600) // 60
        ss = second_int % 60
        file_name = f"{video_path.stem}__t_{hh:02d}{mm:02d}{ss:02d}__{index:03d}.jpg"
        file_path = output_dir / file_name
        cv2.imwrite(str(file_path), frame)

        wall_clock = ""
        if start_dt is not None:
            wall_clock = (start_dt + timedelta(seconds=second)).strftime("%Y-%m-%d %H:%M:%S")

        records.append(
            {
                "image_path": str(file_path.as_posix()),
                "source_video": video_path.name,
                "video_offset_sec": f"{second:.2f}",
                "wall_clock": wall_clock,
                "fps": f"{fps:.2f}",
                "width": str(width),
                "height": str(height),
            }
        )

    cap.release()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deterministic preview frames for article_pack scouting.")
    parser.add_argument("--input-dir", required=True, help="Directory with source videos")
    parser.add_argument("--output-dir", required=True, help="Directory for preview frames")
    parser.add_argument("--count-per-video", type=int, default=30, help="Frames to extract from each video")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No .mp4 files found in {input_dir}")

    manifest_rows = []
    for video_path in videos:
        per_video_dir = output_dir / video_path.stem
        rows = extract_preview_frames(video_path, per_video_dir, args.count_per_video)
        manifest_rows.extend(rows)
        print(f"{video_path.name}: extracted {len(rows)} preview frames")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_path", "source_video", "video_offset_sec", "wall_clock", "fps", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Manifest: {manifest_path}")
    print(f"Total preview frames: {len(manifest_rows)}")


if __name__ == "__main__":
    main()

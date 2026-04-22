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


def time_bucket(video_name: str) -> str | None:
    if "140000" in video_name or "150000" in video_name:
        return "train"
    if "160000" in video_name:
        return "val"
    return None


def offset_timestamps(duration_sec: float, count: int, phase: float) -> list[float]:
    if count <= 0 or duration_sec <= 0:
        return []
    step = duration_sec / count
    values = []
    for i in range(count):
        second = (i + phase) * step
        values.append(min(second, max(duration_sec - 0.001, 0)))
    return values


def extract_for_video(video_path: Path, output_dir: Path, count: int, phase: float) -> list[dict]:
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
    rows = []

    for index, second in enumerate(offset_timestamps(duration_sec, count, phase), start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()
        if not ok:
            continue

        second_int = int(math.floor(second))
        hh = second_int // 3600
        mm = (second_int % 3600) // 60
        ss = second_int % 60
        file_name = f"{video_path.stem}__r2_{hh:02d}{mm:02d}{ss:02d}__{index:03d}.jpg"
        file_path = output_dir / file_name
        cv2.imwrite(str(file_path), frame)

        wall_clock = ""
        if start_dt is not None:
            wall_clock = (start_dt + timedelta(seconds=second)).strftime("%Y-%m-%d %H:%M:%S")

        rows.append(
            {
                "image_path": str(file_path.as_posix()),
                "image_name": file_name,
                "source_video": video_path.name,
                "split": time_bucket(video_path.name),
                "video_offset_sec": f"{second:.2f}",
                "wall_clock": wall_clock,
                "fps": f"{fps:.2f}",
                "width": str(width),
                "height": str(height),
                "phase": f"{phase:.2f}",
            }
        )

    cap.release()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract round2 frames for article_pack expansion.")
    parser.add_argument("--input-dir", required=True, help="Directory with CAM10 source videos")
    parser.add_argument("--output-root", required=True, help="Round2 root directory")
    parser.add_argument("--count-train", type=int, default=30, help="Frames per train video")
    parser.add_argument("--count-val", type=int, default=30, help="Frames for val video")
    parser.add_argument("--phase", type=float, default=0.25, help="Sampling phase to avoid seed_round1 overlap")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    images_root = output_root / "images"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for video_path in sorted(input_dir.glob("*.mp4")):
        split = time_bucket(video_path.name)
        if split is None:
            continue

        count = args.count_train if split == "train" else args.count_val
        out_dir = images_root / split
        rows = extract_for_video(video_path, out_dir, count=count, phase=args.phase)
        all_rows.extend(rows)
        print(f"{video_path.name}: extracted {len(rows)} frames for {split}")

    manifest_path = reports_root / "round2_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image_path",
                "image_name",
                "source_video",
                "split",
                "video_offset_sec",
                "wall_clock",
                "fps",
                "width",
                "height",
                "phase",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    train_count = sum(1 for row in all_rows if row["split"] == "train")
    val_count = sum(1 for row in all_rows if row["split"] == "val")
    print(f"Manifest: {manifest_path}")
    print(f"Round2 counts -> train: {train_count}, val: {val_count}")


if __name__ == "__main__":
    main()

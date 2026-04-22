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


def phased_timestamps(duration_sec: float, count: int, phase: float) -> list[float]:
    if count <= 0 or duration_sec <= 0:
        return []
    step = duration_sec / count
    values = []
    for i in range(count):
        second = (i + phase) * step
        values.append(min(second, max(duration_sec - 0.001, 0)))
    return values


def extract_split(
    video_path: Path,
    output_dir: Path,
    count: int,
    phase: float,
    tag: str,
) -> list[dict]:
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

    for index, second in enumerate(phased_timestamps(duration_sec, count, phase), start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()
        if not ok:
            continue

        second_int = int(math.floor(second))
        hh = second_int // 3600
        mm = (second_int % 3600) // 60
        ss = second_int % 60
        file_name = f"{video_path.stem}__{tag}_{hh:02d}{mm:02d}{ss:02d}__{index:03d}.jpg"
        file_path = output_dir / file_name
        cv2.imwrite(str(file_path), frame)

        wall_clock = ""
        if start_dt is not None:
            wall_clock = (start_dt + timedelta(seconds=second)).strftime("%Y-%m-%d %H:%M:%S")

        rows.append(
            {
                "image_path": str(file_path.resolve().as_posix()),
                "image_name": file_name,
                "source_video": video_path.name,
                "split": tag,
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


def write_path_list(output_path: Path, rows: list[dict]) -> None:
    output_path.write_text("\n".join(row["image_path"] for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fixed test and hard-pool frames from CAM10 17:00 video.")
    parser.add_argument("--video", required=True, help="Path to 17:00 source video")
    parser.add_argument("--output-root", required=True, help="Output root for test_round1")
    parser.add_argument("--test-count", type=int, default=150, help="Number of fixed benchmark test frames")
    parser.add_argument("--hard-count", type=int, default=60, help="Number of hard-pool frames")
    parser.add_argument("--test-phase", type=float, default=0.50, help="Sampling phase for test")
    parser.add_argument("--hard-phase", type=float, default=0.20, help="Sampling phase for hard-pool")
    parser.add_argument("--test-list", required=True, help="Output detection_test.txt path")
    parser.add_argument("--hard-list", required=True, help="Output hard_subset.txt path")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_root = Path(args.output_root)
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    test_rows = extract_split(video_path, output_root / "images" / "test", args.test_count, args.test_phase, "test")
    hard_rows = extract_split(
        video_path, output_root / "images" / "hard_pool", args.hard_count, args.hard_phase, "hard"
    )

    manifest_path = reports_root / "test_round1_manifest.csv"
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
        writer.writerows(test_rows + hard_rows)

    write_path_list(Path(args.test_list), test_rows)
    write_path_list(Path(args.hard_list), hard_rows)

    print(f"Test frames: {len(test_rows)}")
    print(f"Hard-pool frames: {len(hard_rows)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

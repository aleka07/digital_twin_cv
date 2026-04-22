import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def ensure_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


ensure_project_root()

from article_pack.scripts.tracker_pipeline import TrackerPipeline


CLASS_NAMES = {0: "person", 1: "vat"}


@dataclass
class Zone:
    name: str
    points: list[list[int]]
    color: tuple[int, int, int]

    def __post_init__(self) -> None:
        self.polygon = np.array(self.points, dtype=np.int32)

    def contains(self, point: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self.polygon.astype(np.float32), point, False) >= 0


def load_zones(path: Path) -> list[Zone]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Zone(
            name=item["name"],
            points=item["points"],
            color=tuple(item.get("color", [0, 255, 255])),
        )
        for item in raw
    ]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_seconds(seconds: float) -> str:
    whole = int(seconds)
    hh = whole // 3600
    mm = (whole % 3600) // 60
    ss = whole % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def draw_zone_overlay(frame: np.ndarray, zones: list[Zone]) -> np.ndarray:
    output = frame.copy()
    for zone in zones:
        overlay = output.copy()
        cv2.fillPoly(overlay, [zone.polygon], zone.color)
        output = cv2.addWeighted(overlay, 0.14, output, 0.86, 0)
        cv2.polylines(output, [zone.polygon], True, zone.color, 3, cv2.LINE_AA)
        cx = int(np.mean(zone.polygon[:, 0]))
        cy = int(np.mean(zone.polygon[:, 1]))
        cv2.putText(output, zone.name, (cx - 90, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def save_preview(
    output_path: Path,
    frame: np.ndarray,
    zones: list[Zone],
    tracker: TrackerPipeline,
    detections: list,
    timestamp_text: str,
) -> None:
    drawn = draw_zone_overlay(frame, zones)
    drawn = tracker.draw(drawn, detections, show_conf=False, show_track_id=True, thickness=3)
    cv2.rectangle(drawn, (20, 20), (420, 74), (0, 0, 0), -1)
    cv2.putText(drawn, timestamp_text, (34, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2, cv2.LINE_AA)
    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), drawn)


def write_markdown(
    path: Path,
    video_path: Path,
    model_path: str,
    tracker_name: str,
    zones_path: Path,
    fps: float,
    frames_processed: int,
    summary_rows: list[dict],
    preview_names: list[str],
) -> None:
    duration = frames_processed / fps if fps else 0.0
    lines = ["# Long Zone Demo Summary", ""]
    lines.append(f"- video: `{video_path}`")
    lines.append(f"- detector: `{model_path}`")
    lines.append(f"- tracker: `{tracker_name}`")
    lines.append(f"- zones: `{zones_path}`")
    lines.append(f"- fps: `{fps:.3f}`")
    lines.append(f"- frames_processed: `{frames_processed}`")
    lines.append(f"- duration: `{duration / 60:.2f} min` (`{format_seconds(duration)}`)")
    lines.append("- membership rule: bbox center inside polygon")
    lines.append("")
    lines.append("## Zone Summary")
    lines.append("")
    for row in summary_rows:
        lines.append(
            f"- `{row['zone']}`: entries={row['entries_total']}, exits={row['exits_total']}, "
            f"avg_occ={row['avg_occupancy']}, max_occ={row['max_occupancy']}, "
            f"total_dwell_s={row['total_dwell_seconds']}, occupied_pct={row['occupied_pct']}"
        )
    if preview_names:
        lines.append("")
        lines.append("## Preview Frames")
        lines.append("")
        for name in preview_names:
            lines.append(f"- `{name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a long detector+tracker+zone analytics demo on a CAM10 video.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--zones", required=True, help="Path to zone JSON.")
    parser.add_argument("--model", required=True, help="Detector checkpoint path.")
    parser.add_argument("--tracker", default="bytetrack", help="Tracker name or YAML path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the demo run.")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start offset inside the video.")
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="Process this many seconds; 0 means until end.")
    parser.add_argument("--conf", type=float, default=0.25, help="Tracker confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="Tracker IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--preview-count", type=int, default=6, help="How many preview frames to save across the run.")
    args = parser.parse_args()

    video_path = Path(args.video)
    zones_path = Path(args.zones)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    preview_dir = output_dir / "previews"
    ensure_dir(preview_dir)

    zones = load_zones(zones_path)
    tracker = TrackerPipeline(
        model_path=args.model,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(args.start_seconds * fps))) if fps else 0
    end_frame = total_frames
    if args.duration_seconds and fps:
        end_frame = min(total_frames, start_frame + int(round(args.duration_seconds * fps)))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_to_process = max(0, end_frame - start_frame)
    preview_points = set()
    if frames_to_process > 0 and args.preview_count > 0:
        for i in range(args.preview_count):
            rel = int(round(i * (frames_to_process - 1) / max(1, args.preview_count - 1)))
            preview_points.add(rel)

    dwell_frames: Counter = Counter()
    prev_inside: dict[tuple[str, int], bool] = defaultdict(bool)
    zone_entries_exits: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: {"entries": 0, "exits": 0})
    occupancy_rows: list[dict] = []
    event_rows: list[dict] = []
    preview_names: list[str] = []
    unique_track_ids: set[int] = set()

    frame_idx = start_frame
    rel_idx = 0

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        detections = tracker.process_frame(frame)
        zone_frame_counts = Counter()
        zone_frame_class_counts = Counter()

        for det in detections:
            unique_track_ids.add(det.track_id)
            center = det.center
            class_id = int(det.cls)
            if class_id not in CLASS_NAMES:
                continue
            for zone in zones:
                inside = zone.contains(center)
                key = (zone.name, det.track_id)
                if inside:
                    dwell_frames[(zone.name, det.track_id, class_id)] += 1
                    zone_frame_counts[zone.name] += 1
                    zone_frame_class_counts[(zone.name, class_id)] += 1
                was_inside = prev_inside[key]
                if inside and not was_inside:
                    zone_entries_exits[(zone.name, class_id)]["entries"] += 1
                    event_rows.append(
                        {
                            "frame_id": frame_idx + 1,
                            "timestamp_seconds": round(frame_idx / fps, 3) if fps else 0.0,
                            "timestamp_hms": format_seconds(frame_idx / fps) if fps else "00:00:00",
                            "event": "entry",
                            "zone": zone.name,
                            "track_id": det.track_id,
                            "class_id": class_id,
                            "class_name": CLASS_NAMES[class_id],
                            "x_center": center[0],
                            "y_center": center[1],
                        }
                    )
                elif not inside and was_inside:
                    zone_entries_exits[(zone.name, class_id)]["exits"] += 1
                    event_rows.append(
                        {
                            "frame_id": frame_idx + 1,
                            "timestamp_seconds": round(frame_idx / fps, 3) if fps else 0.0,
                            "timestamp_hms": format_seconds(frame_idx / fps) if fps else "00:00:00",
                            "event": "exit",
                            "zone": zone.name,
                            "track_id": det.track_id,
                            "class_id": class_id,
                            "class_name": CLASS_NAMES[class_id],
                            "x_center": center[0],
                            "y_center": center[1],
                        }
                    )
                prev_inside[key] = inside

        occ_row = {
            "frame_id": frame_idx + 1,
            "relative_frame": rel_idx + 1,
            "timestamp_seconds": round(frame_idx / fps, 3) if fps else 0.0,
            "timestamp_hms": format_seconds(frame_idx / fps) if fps else "00:00:00",
        }
        for zone in zones:
            zone_name = zone.name
            occ_row[f"{zone_name}_total"] = zone_frame_counts[zone_name]
            occ_row[f"{zone_name}_person"] = zone_frame_class_counts[(zone_name, 0)]
            occ_row[f"{zone_name}_vat"] = zone_frame_class_counts[(zone_name, 1)]
        occupancy_rows.append(occ_row)

        if rel_idx in preview_points:
            timestamp_text = format_seconds(frame_idx / fps) if fps else f"frame {frame_idx + 1}"
            preview_name = f"preview_{rel_idx + 1:06d}.jpg"
            save_preview(preview_dir / preview_name, frame, zones, tracker, detections, timestamp_text)
            preview_names.append(preview_name)

        rel_idx += 1
        frame_idx += 1

    cap.release()

    dwell_rows = []
    for (zone_name, track_id, class_id), frames in sorted(dwell_frames.items()):
        dwell_rows.append(
            {
                "zone": zone_name,
                "track_id": track_id,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "dwell_frames": frames,
                "dwell_seconds": round(frames / fps, 3) if fps else 0.0,
            }
        )

    summary_rows = []
    for zone in zones:
        zone_name = zone.name
        total_dwell = sum(row["dwell_seconds"] for row in dwell_rows if row["zone"] == zone_name)
        occupied_frames = sum(1 for row in occupancy_rows if row[f"{zone_name}_total"] > 0)
        avg_occ = float(np.mean([row[f"{zone_name}_total"] for row in occupancy_rows])) if occupancy_rows else 0.0
        max_occ = max((row[f"{zone_name}_total"] for row in occupancy_rows), default=0)
        summary_rows.append(
            {
                "zone": zone_name,
                "entries_total": sum(zone_entries_exits[(zone_name, cls)]["entries"] for cls in CLASS_NAMES),
                "exits_total": sum(zone_entries_exits[(zone_name, cls)]["exits"] for cls in CLASS_NAMES),
                "entries_person": zone_entries_exits[(zone_name, 0)]["entries"],
                "exits_person": zone_entries_exits[(zone_name, 0)]["exits"],
                "entries_vat": zone_entries_exits[(zone_name, 1)]["entries"],
                "exits_vat": zone_entries_exits[(zone_name, 1)]["exits"],
                "avg_occupancy": round(avg_occ, 3),
                "max_occupancy": int(max_occ),
                "occupied_frames": occupied_frames,
                "occupied_pct": round((occupied_frames / len(occupancy_rows) * 100.0), 2) if occupancy_rows else 0.0,
                "total_dwell_seconds": round(total_dwell, 3),
            }
        )

    meta = {
        "video": str(video_path),
        "zones": str(zones_path),
        "model": args.model,
        "tracker": args.tracker,
        "fps": fps,
        "total_frames_in_video": total_frames,
        "start_frame": start_frame + 1,
        "end_frame": start_frame + len(occupancy_rows),
        "frames_processed": len(occupancy_rows),
        "duration_seconds": round(len(occupancy_rows) / fps, 3) if fps else 0.0,
        "duration_hms": format_seconds(len(occupancy_rows) / fps) if fps else "00:00:00",
        "unique_track_ids": len([track_id for track_id in unique_track_ids if track_id >= 0]),
    }

    write_csv(
        output_dir / "zone_summary.csv",
        summary_rows,
        [
            "zone",
            "entries_total",
            "exits_total",
            "entries_person",
            "exits_person",
            "entries_vat",
            "exits_vat",
            "avg_occupancy",
            "max_occupancy",
            "occupied_frames",
            "occupied_pct",
            "total_dwell_seconds",
        ],
    )
    write_csv(
        output_dir / "zone_dwell_by_track.csv",
        dwell_rows,
        ["zone", "track_id", "class_id", "class_name", "dwell_frames", "dwell_seconds"],
    )
    write_csv(
        output_dir / "zone_event_log.csv",
        event_rows,
        ["frame_id", "timestamp_seconds", "timestamp_hms", "event", "zone", "track_id", "class_id", "class_name", "x_center", "y_center"],
    )
    write_csv(
        output_dir / "zone_occupancy_timeline.csv",
        occupancy_rows,
        list(occupancy_rows[0].keys()) if occupancy_rows else ["frame_id", "relative_frame", "timestamp_seconds", "timestamp_hms"],
    )
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_markdown(
        output_dir / "zone_demo_summary.md",
        video_path,
        args.model,
        args.tracker,
        zones_path,
        fps,
        len(occupancy_rows),
        summary_rows,
        preview_names,
    )

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

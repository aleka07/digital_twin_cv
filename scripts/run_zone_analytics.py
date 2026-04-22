import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {0: "person", 1: "vat"}
TRACK_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 128),
]


@dataclass
class TrackDet:
    frame_id: int
    track_id: int
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class Zone:
    def __init__(self, name: str, points: list[list[int]], color: list[int] | None = None):
        self.name = name
        self.points = points
        self.color = tuple(color or [0, 255, 255])
        self._polygon = np.array(points, dtype=np.int32)

    def contains(self, point: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self._polygon.astype(np.float32), point, False) >= 0


def ensure_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def parse_tracks(path: Path) -> dict[int, list[TrackDet]]:
    by_frame: dict[int, list[TrackDet]] = defaultdict(list)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        frame_id, track_id, class_id, x1, y1, x2, y2 = map(int, parts)
        by_frame[frame_id].append(TrackDet(frame_id, track_id, class_id, x1, y1, x2, y2))
    return dict(by_frame)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_zones(path: Path) -> list[Zone]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Zone(item["name"], item["points"], item.get("color")) for item in raw]


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def run_clip_analysis(clip_dir: Path, zones: list[Zone]) -> dict[str, object]:
    manifest = read_manifest(clip_dir / "frame_manifest.csv")
    tracks_by_frame = parse_tracks(clip_dir / "gt_tracks.txt")
    fps = float(manifest[0]["fps"]) if manifest else 1.0

    dwell_frames: dict[tuple[str, int, int], int] = Counter()
    prev_inside: dict[tuple[str, int], bool] = defaultdict(bool)
    entry_exit: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: {"entries": 0, "exits": 0})
    occupancy_per_zone: dict[str, list[int]] = defaultdict(list)
    occupancy_per_zone_class: dict[tuple[str, int], list[int]] = defaultdict(list)
    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for row in manifest:
        frame_id = int(row["frame_id"])
        detections = tracks_by_frame.get(frame_id, [])
        zone_frame_counts = Counter()
        zone_frame_class_counts = Counter()

        for det in detections:
            center = det.center
            trails[det.track_id].append(center)
            for zone in zones:
                inside = zone.contains(center)
                key = (zone.name, det.track_id)
                if inside:
                    dwell_frames[(zone.name, det.track_id, det.class_id)] += 1
                    zone_frame_counts[zone.name] += 1
                    zone_frame_class_counts[(zone.name, det.class_id)] += 1
                was_inside = prev_inside[key]
                if inside and not was_inside:
                    entry_exit[(zone.name, det.class_id)]["entries"] += 1
                elif not inside and was_inside:
                    entry_exit[(zone.name, det.class_id)]["exits"] += 1
                prev_inside[key] = inside

        for zone in zones:
            occupancy_per_zone[zone.name].append(zone_frame_counts[zone.name])
            for class_id in CLASS_NAMES:
                occupancy_per_zone_class[(zone.name, class_id)].append(zone_frame_class_counts[(zone.name, class_id)])

    dwell_rows = []
    for (zone_name, track_id, class_id), frames in sorted(dwell_frames.items()):
        dwell_rows.append(
            {
                "clip": clip_dir.name,
                "zone": zone_name,
                "track_id": track_id,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "dwell_frames": frames,
                "dwell_seconds": round(frames / fps, 3),
            }
        )

    summary_rows = []
    for zone in zones:
        zone_name = zone.name
        entry_total = sum(entry_exit[(zone_name, cls)]["entries"] for cls in CLASS_NAMES)
        exit_total = sum(entry_exit[(zone_name, cls)]["exits"] for cls in CLASS_NAMES)
        avg_occ = np.mean(occupancy_per_zone[zone_name]) if occupancy_per_zone[zone_name] else 0.0
        max_occ = max(occupancy_per_zone[zone_name]) if occupancy_per_zone[zone_name] else 0
        occupied_frames = sum(1 for value in occupancy_per_zone[zone_name] if value > 0)
        total_dwell = sum(row["dwell_seconds"] for row in dwell_rows if row["zone"] == zone_name)

        summary_rows.append(
            {
                "clip": clip_dir.name,
                "zone": zone_name,
                "entries_total": entry_total,
                "exits_total": exit_total,
                "entries_person": entry_exit[(zone_name, 0)]["entries"],
                "exits_person": entry_exit[(zone_name, 0)]["exits"],
                "entries_vat": entry_exit[(zone_name, 1)]["entries"],
                "exits_vat": entry_exit[(zone_name, 1)]["exits"],
                "avg_occupancy": round(float(avg_occ), 3),
                "max_occupancy": int(max_occ),
                "occupied_frames": occupied_frames,
                "total_dwell_seconds": round(total_dwell, 3),
                "avg_person_occupancy": round(float(np.mean(occupancy_per_zone_class[(zone_name, 0)])), 3),
                "avg_vat_occupancy": round(float(np.mean(occupancy_per_zone_class[(zone_name, 1)])), 3),
            }
        )

    return {
        "fps": fps,
        "dwell_rows": dwell_rows,
        "summary_rows": summary_rows,
        "trails": trails,
        "manifest": manifest,
        "tracks_by_frame": tracks_by_frame,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_zone_preview(image_path: Path, zones: list[Zone], trails: dict[int, list[tuple[int, int]]], output_path: Path) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return

    for zone in zones:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(zone.points, dtype=np.int32)], zone.color)
        frame = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0)
        cv2.polylines(frame, [np.array(zone.points, dtype=np.int32)], True, zone.color, 3, cv2.LINE_AA)
        cx = int(np.mean(np.array(zone.points)[:, 0]))
        cy = int(np.mean(np.array(zone.points)[:, 1]))
        cv2.putText(frame, zone.name, (cx - 80, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    for track_id, trail in trails.items():
        if len(trail) < 2:
            continue
        color = color_for_track(track_id)
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], color, 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def write_markdown(path: Path, zones_path: Path, summary_rows: list[dict], dwell_rows: list[dict], clips: list[str]) -> None:
    lines = ["# Zone Analytics Summary", ""]
    lines.append(f"- zones: `{zones_path}`")
    lines.append(f"- clips: `{', '.join(clips)}`")
    lines.append("- membership rule: bbox center inside polygon")
    lines.append("")

    per_clip = defaultdict(list)
    for row in summary_rows:
        per_clip[row["clip"]].append(row)

    for clip_name in clips:
        lines.append(f"## {clip_name}")
        clip_rows = per_clip[clip_name]
        for row in clip_rows:
            lines.append(
                f"- `{row['zone']}`: entries={row['entries_total']}, exits={row['exits_total']}, "
                f"avg_occ={row['avg_occupancy']}, max_occ={row['max_occupancy']}, total_dwell_s={row['total_dwell_seconds']}"
            )
        top_dwell = sorted([row for row in dwell_rows if row["clip"] == clip_name], key=lambda r: r["dwell_seconds"], reverse=True)[:5]
        if top_dwell:
            lines.append("- top dwell tracks:")
            for row in top_dwell:
                lines.append(
                    f"  - zone=`{row['zone']}` track=`{row['track_id']}` class=`{row['class_name']}` dwell_s={row['dwell_seconds']}"
                )
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zone analytics on article_pack tracking GT clips.")
    parser.add_argument(
        "--tracking-root",
        default="article_pack/tracking_benchmark",
        help="Tracking benchmark root",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        default=["clip_A", "clip_B"],
        help="Clip names to analyze",
    )
    parser.add_argument(
        "--zones",
        default="article_pack/zones/cam10_zones_final.json",
        help="Zones JSON path",
    )
    parser.add_argument(
        "--summary-csv",
        default="article_pack/results/zone_summary.csv",
        help="Output zone summary CSV",
    )
    parser.add_argument(
        "--dwell-csv",
        default="article_pack/results/zone_dwell_by_track.csv",
        help="Output dwell-by-track CSV",
    )
    parser.add_argument(
        "--summary-md",
        default="article_pack/results/zone_analytics_summary.md",
        help="Output markdown summary",
    )
    parser.add_argument(
        "--preview-dir",
        default="article_pack/results/zone_previews",
        help="Directory for preview overlays",
    )
    args = parser.parse_args()

    ensure_project_root()
    zones_path = Path(args.zones)
    zones = load_zones(zones_path)
    tracking_root = Path(args.tracking_root)

    all_summary_rows: list[dict] = []
    all_dwell_rows: list[dict] = []

    for clip_name in args.clips:
        clip_dir = tracking_root / clip_name
        result = run_clip_analysis(clip_dir, zones)
        all_summary_rows.extend(result["summary_rows"])
        all_dwell_rows.extend(result["dwell_rows"])
        first_image = clip_dir / "frames" / result["manifest"][0]["image_name"]
        preview_path = Path(args.preview_dir) / f"{clip_name}_zones_trails.jpg"
        render_zone_preview(first_image, zones, result["trails"], preview_path)
        print(f"{clip_name}: analyzed {len(result['manifest'])} frames, preview={preview_path}")

    write_csv(
        Path(args.summary_csv),
        all_summary_rows,
        [
            "clip",
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
            "total_dwell_seconds",
            "avg_person_occupancy",
            "avg_vat_occupancy",
        ],
    )
    write_csv(
        Path(args.dwell_csv),
        all_dwell_rows,
        ["clip", "zone", "track_id", "class_id", "class_name", "dwell_frames", "dwell_seconds"],
    )
    write_markdown(Path(args.summary_md), zones_path, all_summary_rows, all_dwell_rows, args.clips)
    print(f"Saved: {args.summary_csv}")
    print(f"Saved: {args.dwell_csv}")
    print(f"Saved: {args.summary_md}")


if __name__ == "__main__":
    main()

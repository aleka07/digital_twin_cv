import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

from trackeval import metrics


def ensure_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def parse_track_file(path: Path) -> dict[int, list[dict[str, int]]]:
    by_frame: dict[int, list[dict[str, int]]] = {}
    if not path.exists():
        return by_frame

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        frame_id, track_id, class_id, x1, y1, x2, y2 = map(int, parts)
        by_frame.setdefault(frame_id, []).append(
            {
                "track_id": track_id,
                "class_id": class_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return by_frame


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def box_iou(a: dict[str, int], b: dict[str, int]) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_trackeval_data(gt_by_frame: dict[int, list[dict[str, int]]], pred_by_frame: dict[int, list[dict[str, int]]], num_frames: int) -> dict:
    gt_ids_sorted = sorted({box["track_id"] for boxes in gt_by_frame.values() for box in boxes})
    pred_ids_sorted = sorted({box["track_id"] for boxes in pred_by_frame.values() for box in boxes})
    gt_id_map = {track_id: idx for idx, track_id in enumerate(gt_ids_sorted)}
    pred_id_map = {track_id: idx for idx, track_id in enumerate(pred_ids_sorted)}

    data = {
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(pred_id_map),
        "num_gt_dets": sum(len(boxes) for boxes in gt_by_frame.values()),
        "num_tracker_dets": sum(len(boxes) for boxes in pred_by_frame.values()),
        "num_timesteps": num_frames,
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
    }

    for frame_id in range(1, num_frames + 1):
        gt_boxes = gt_by_frame.get(frame_id, [])
        pred_boxes = pred_by_frame.get(frame_id, [])
        gt_ids = np.array([gt_id_map[box["track_id"]] for box in gt_boxes], dtype=int)
        pred_ids = np.array([pred_id_map[box["track_id"]] for box in pred_boxes], dtype=int)
        sim = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
        for gi, gt_box in enumerate(gt_boxes):
            for pi, pred_box in enumerate(pred_boxes):
                if gt_box["class_id"] != pred_box["class_id"]:
                    continue
                sim[gi, pi] = box_iou(gt_box, pred_box)
        data["gt_ids"].append(gt_ids)
        data["tracker_ids"].append(pred_ids)
        data["similarity_scores"].append(sim)

    return data


def scalar_hota(value: np.ndarray | float) -> float:
    if isinstance(value, np.ndarray):
        return float(np.mean(value))
    return float(value)


def evaluate_sequence(gt_path: Path, pred_path: Path, manifest_path: Path) -> dict[str, float]:
    gt_by_frame = parse_track_file(gt_path)
    pred_by_frame = parse_track_file(pred_path)
    num_frames = len(read_manifest(manifest_path))
    data = build_trackeval_data(gt_by_frame, pred_by_frame, num_frames)

    clear_metric = metrics.CLEAR({"PRINT_CONFIG": False})
    identity_metric = metrics.Identity({"PRINT_CONFIG": False})
    hota_metric = metrics.HOTA({"PRINT_CONFIG": False})

    clear_res = clear_metric.eval_sequence(data)
    identity_res = identity_metric.eval_sequence(data)
    hota_res = hota_metric.eval_sequence(data)

    return {
        "mota": float(clear_res["MOTA"]),
        "id_switches": int(clear_res["IDSW"]),
        "idf1": float(identity_res["IDF1"]),
        "hota": scalar_hota(hota_res["HOTA"]),
        "num_frames": num_frames,
    }


def combine_results(per_clip: dict[str, dict[str, float]]) -> dict[str, float]:
    clear_metric = metrics.CLEAR({"PRINT_CONFIG": False})
    identity_metric = metrics.Identity({"PRINT_CONFIG": False})
    hota_metric = metrics.HOTA({"PRINT_CONFIG": False})

    clear_inputs = {}
    identity_inputs = {}
    hota_inputs = {}
    for clip_name, res in per_clip.items():
        clear_inputs[clip_name] = {
            "CLR_TP": res["clear"]["CLR_TP"],
            "CLR_FN": res["clear"]["CLR_FN"],
            "CLR_FP": res["clear"]["CLR_FP"],
            "IDSW": res["clear"]["IDSW"],
            "MT": res["clear"]["MT"],
            "PT": res["clear"]["PT"],
            "ML": res["clear"]["ML"],
            "Frag": res["clear"]["Frag"],
            "MOTP_sum": res["clear"]["MOTP_sum"],
            "CLR_Frames": res["clear"]["CLR_Frames"],
        }
        identity_inputs[clip_name] = {
            "IDTP": res["identity"]["IDTP"],
            "IDFN": res["identity"]["IDFN"],
            "IDFP": res["identity"]["IDFP"],
        }
        hota_inputs[clip_name] = res["hota_raw"]

    clear_combined = clear_metric.combine_sequences(clear_inputs)
    identity_combined = identity_metric.combine_sequences(identity_inputs)
    hota_combined = hota_metric.combine_sequences(hota_inputs)
    return {
        "mota": float(clear_combined["MOTA"]),
        "id_switches": int(clear_combined["IDSW"]),
        "idf1": float(identity_combined["IDF1"]),
        "hota": scalar_hota(hota_combined["HOTA"]),
    }


def evaluate_sequence_full(gt_path: Path, pred_path: Path, manifest_path: Path) -> dict[str, object]:
    gt_by_frame = parse_track_file(gt_path)
    pred_by_frame = parse_track_file(pred_path)
    num_frames = len(read_manifest(manifest_path))
    data = build_trackeval_data(gt_by_frame, pred_by_frame, num_frames)

    clear_metric = metrics.CLEAR({"PRINT_CONFIG": False})
    identity_metric = metrics.Identity({"PRINT_CONFIG": False})
    hota_metric = metrics.HOTA({"PRINT_CONFIG": False})

    clear_res = clear_metric.eval_sequence(data)
    identity_res = identity_metric.eval_sequence(data)
    hota_res = hota_metric.eval_sequence(data)

    return {
        "summary": {
            "mota": float(clear_res["MOTA"]),
            "id_switches": int(clear_res["IDSW"]),
            "idf1": float(identity_res["IDF1"]),
            "hota": scalar_hota(hota_res["HOTA"]),
            "num_frames": num_frames,
        },
        "clear": clear_res,
        "identity": identity_res,
        "hota_raw": hota_res,
    }


def measure_tracker_fps(model_path: str, tracker_name: str, tracking_root: Path, clips: list[str], imgsz: int, conf: float, iou: float) -> float:
    ensure_project_root()
    from article_pack.scripts.tracker_pipeline import TrackerPipeline

    pipe = TrackerPipeline(
        model_path=model_path,
        tracker=tracker_name,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
    )

    total_frames = 0
    started = time.perf_counter()
    for clip_name in clips:
        frames_dir = tracking_root / clip_name / "frames"
        frames = sorted(path for path in frames_dir.glob("*.jpg") if path.is_file())
        pipe.reset()
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            pipe.process_frame(frame)
            total_frames += 1
    elapsed = max(1e-9, time.perf_counter() - started)
    return total_frames / elapsed


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["tracker", "detector", "hota", "mota", "idf1", "id_switches", "fps", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_details(path: Path, tracker_results: dict[str, dict[str, object]]) -> None:
    lines = ["# Tracking Benchmark Details", ""]
    for tracker_name, payload in tracker_results.items():
        combined = payload["combined"]
        lines.append(f"## {tracker_name}")
        lines.append(
            f"- combined: HOTA={combined['hota']:.3f}, MOTA={combined['mota']:.3f}, IDF1={combined['idf1']:.3f}, IDSW={combined['id_switches']}"
        )
        lines.append(f"- fps: {payload['fps']:.1f}")
        lines.append("- per clip:")
        for clip_name, clip_res in payload["clips"].items():
            summary = clip_res["summary"]
            lines.append(
                f"  - {clip_name}: HOTA={summary['hota']:.3f}, MOTA={summary['mota']:.3f}, "
                f"IDF1={summary['idf1']:.3f}, IDSW={summary['id_switches']}"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ByteTrack and BoT-SORT drafts against article_pack GT clips.")
    parser.add_argument(
        "--tracking-root",
        default="article_pack/tracking_benchmark",
        help="Tracking benchmark root",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        default=["clip_A", "clip_B"],
        help="Clip names to evaluate",
    )
    parser.add_argument(
        "--model",
        default="runs/detect/article_pack/experiments/detector/cam10_round12_yolo11x/weights/best.pt",
        help="Detector weights used for runtime measurement",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Tracking detection confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="Tracking NMS IoU")
    parser.add_argument(
        "--output-csv",
        default="article_pack/results/tracker_benchmark.csv",
        help="Output benchmark CSV",
    )
    parser.add_argument(
        "--details-md",
        default="article_pack/results/tracking_benchmark_details.md",
        help="Detailed markdown report path",
    )
    parser.add_argument(
        "--tracker-spec",
        action="append",
        default=None,
        help="Tracker spec in the form label=prediction_file. Can be repeated.",
    )
    parser.add_argument(
        "--runtime-tracker-spec",
        action="append",
        default=None,
        help="Runtime tracker spec in the form label=tracker_name_or_yaml. Can be repeated.",
    )
    args = parser.parse_args()

    tracking_root = Path(args.tracking_root)
    output_csv = Path(args.output_csv)
    details_md = Path(args.details_md)
    tracker_specs = args.tracker_spec or ["bytetrack=draft_bytetrack.txt", "botsort=draft_botsort.txt"]
    runtime_specs_raw = args.runtime_tracker_spec or []
    runtime_specs: dict[str, str] = {}
    for item in runtime_specs_raw:
        label, tracker_value = item.split("=", 1)
        runtime_specs[label] = tracker_value

    tracker_results: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, str]] = []

    for spec in tracker_specs:
        tracker_name, pred_file = spec.split("=", 1)
        clip_payload: dict[str, object] = {}
        for clip_name in args.clips:
            clip_dir = tracking_root / clip_name
            clip_payload[clip_name] = evaluate_sequence_full(
                clip_dir / "gt_tracks.txt",
                clip_dir / pred_file,
                clip_dir / "frame_manifest.csv",
            )

        combined = combine_results(clip_payload)
        fps = measure_tracker_fps(
            model_path=args.model,
            tracker_name=runtime_specs.get(tracker_name, tracker_name),
            tracking_root=tracking_root,
            clips=args.clips,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )

        tracker_results[tracker_name] = {
            "combined": combined,
            "clips": clip_payload,
            "fps": fps,
        }

        notes = f"clips={'+'.join(args.clips)}, iou=0.5, conf={args.conf}, imgsz={args.imgsz}"
        csv_rows.append(
            {
                "tracker": tracker_name,
                "detector": Path(args.model).name,
                "hota": f"{combined['hota']:.3f}",
                "mota": f"{combined['mota']:.3f}",
                "idf1": f"{combined['idf1']:.3f}",
                "id_switches": str(combined["id_switches"]),
                "fps": f"{fps:.1f}",
                "notes": notes,
            }
        )

        print(
            f"{tracker_name}: HOTA={combined['hota']:.3f} MOTA={combined['mota']:.3f} "
            f"IDF1={combined['idf1']:.3f} IDSW={combined['id_switches']} FPS={fps:.1f}"
        )

    write_csv(output_csv, csv_rows)
    write_details(details_md, tracker_results)


if __name__ == "__main__":
    main()

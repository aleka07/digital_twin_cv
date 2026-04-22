import argparse
import csv
from pathlib import Path

import cv2


CLASS_NAMES = {0: "person", 1: "vat"}
COCO_TO_CUSTOM = {0: 0}
COCO_VAT_CANDIDATES = {24, 25, 26, 27, 39, 41, 42, 43, 44, 45, 46}


def auto_label_dir(images_dir: Path, labels_dir: Path, model_path: str, conf: float, skip_existing: bool) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    labels_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(model.names) if hasattr(model, "names") else 80
    is_finetuned = num_classes <= len(CLASS_NAMES)

    image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    total_detections = 0
    non_empty = 0
    rows = []

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if skip_existing and label_path.exists():
            rows.append(
                {
                    "image_name": img_path.name,
                    "label_name": label_path.name,
                    "detections": "preserved",
                    "status": "kept_existing",
                }
            )
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model(frame, conf=conf, verbose=False)
        result = results[0]

        lines = []
        if result.boxes is not None:
            for idx in range(len(result.boxes)):
                det_cls = int(result.boxes.cls[idx])
                x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy()

                if is_finetuned:
                    our_cls = det_cls
                else:
                    if det_cls in COCO_TO_CUSTOM:
                        our_cls = COCO_TO_CUSTOM[det_cls]
                    elif det_cls in COCO_VAT_CANDIDATES:
                        our_cls = 1
                    else:
                        continue

                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text("\n".join(lines), encoding="utf-8")

        total_detections += len(lines)
        if lines:
            non_empty += 1

        rows.append(
            {
                "image_name": img_path.name,
                "label_name": label_path.name,
                "detections": len(lines),
                "status": "auto_labeled",
            }
        )

    return {
        "images": len(image_files),
        "non_empty": non_empty,
        "detections": total_detections,
        "rows": rows,
        "model_classes": num_classes,
        "mode": "fine_tuned" if is_finetuned else "coco_mapped",
    }


def write_report(report_path: Path, summary: dict, split: str, model_path: str, conf: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "image_name", "label_name", "detections", "status"])
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({"split": split, **row})

    summary_path = report_path.with_name(report_path.stem + "_summary.txt")
    summary_path.write_text(
        "\n".join(
            [
                f"split={split}",
                f"model={model_path}",
                f"conf={conf}",
                f"model_mode={summary['mode']}",
                f"model_classes={summary['model_classes']}",
                f"images={summary['images']}",
                f"non_empty={summary['non_empty']}",
                f"detections={summary['detections']}",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label seed round images with YOLO11x.")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--labels", required=True, help="Output labels directory")
    parser.add_argument("--report", required=True, help="CSV report path")
    parser.add_argument("--model", default="yolo11x.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--split", required=True, help="Split name for reporting")
    parser.add_argument("--skip-existing", action="store_true", help="Keep existing label files and only fill missing ones")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    report_path = Path(args.report)

    summary = auto_label_dir(images_dir, labels_dir, args.model, args.conf, args.skip_existing)
    write_report(report_path, summary, args.split, args.model, args.conf)

    print(f"Auto-label complete for {args.split}")
    print(f"  images: {summary['images']}")
    print(f"  non-empty: {summary['non_empty']}")
    print(f"  detections: {summary['detections']}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()

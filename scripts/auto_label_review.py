import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


CLASS_NAMES = {0: "person", 1: "vat"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 165, 0)}
COCO_TO_CUSTOM = {0: 0}
COCO_VAT_CANDIDATES = {24, 25, 26, 27, 39, 41, 42, 43, 44, 45, 46}


def auto_label_all(
    images_dir: Path,
    labels_dir: Path,
    model_path: str,
    conf: float = 0.15,
    include_vat_candidates: bool = True,
) -> None:
    from ultralytics import YOLO

    model = YOLO(model_path)
    labels_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(model.names) if hasattr(model, "names") else 80
    is_finetuned = num_classes <= len(CLASS_NAMES)
    image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))

    print(f"Auto-labeling {len(image_files)} images with {model_path} (conf={conf})...")
    total_detections = 0

    for i, img_path in enumerate(image_files, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model(frame, conf=conf, verbose=False)
        r = results[0]

        lines = []
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                det_cls = int(r.boxes.cls[j])
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()

                if is_finetuned:
                    our_cls = det_cls
                else:
                    if det_cls in COCO_TO_CUSTOM:
                        our_cls = COCO_TO_CUSTOM[det_cls]
                    elif include_vat_candidates and det_cls in COCO_VAT_CANDIDATES:
                        our_cls = 1
                    else:
                        continue

                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = labels_dir / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        total_detections += len(lines)

        if i % 20 == 0 or i == len(image_files):
            print(f"  [{i}/{len(image_files)}] {total_detections} total detections")

    print(f"\nSaved labels to {labels_dir}")


def load_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    labels = []
    if not label_path.exists():
        return labels

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return labels

    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        labels.append((cls, cx, cy, bw, bh))
    return labels


def save_labels(label_path: Path, labels: List[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in labels]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def draw_labels(
    frame: np.ndarray,
    labels: list,
    selected_idx: int = -1,
    box_thickness: int = 4,
    selected_thickness: int = 7,
) -> np.ndarray:
    display = frame.copy()
    h, w = frame.shape[:2]

    for i, (cls, cx, cy, bw, bh) in enumerate(labels):
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = CLASS_COLORS.get(cls, (128, 128, 128))
        thickness = selected_thickness if i == selected_idx else box_thickness
        if i == selected_idx:
            color = (0, 0, 255)

        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        name = CLASS_NAMES.get(cls, f"cls{cls}")
        label = f"[{i}] {name}"
        cv2.putText(display, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    return display


def quick_finetune(
    images_dir: Path,
    labels_dir: Path,
    base_model: str,
    reviewed_indices: set,
    image_files: list,
    epochs: int = 10,
) -> str:
    import shutil
    import yaml
    from ultralytics import YOLO

    tmp_dir = Path("article_pack") / "_active_learning_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_train_img = tmp_dir / "images" / "train"
    tmp_train_lbl = tmp_dir / "labels" / "train"
    tmp_train_img.mkdir(parents=True, exist_ok=True)
    tmp_train_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx in reviewed_indices:
        if idx >= len(image_files):
            continue
        img_path = image_files[idx]
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not img_path.exists() or not lbl_path.exists():
            continue

        shutil.copy2(img_path, tmp_train_img / img_path.name)
        clean_lines = []
        text = lbl_path.read_text(encoding="utf-8").strip()
        if text:
            for line in text.splitlines():
                cls_id = int(line.split()[0])
                if cls_id <= 1:
                    clean_lines.append(line)
        (tmp_train_lbl / lbl_path.name).write_text("\n".join(clean_lines), encoding="utf-8")
        count += 1

    if count < 5:
        print(f"Only {count} reviewed images available. Need at least 5 before quick fine-tune.")
        return base_model

    data_yaml = tmp_dir / "data.yaml"
    data_cfg = {
        "path": str(tmp_dir.resolve()),
        "train": "images/train",
        "val": "images/train",
        "names": {0: "person", 1: "vat"},
    }
    data_yaml.write_text(yaml.dump(data_cfg), encoding="utf-8")

    print(f"\nQuick fine-tune on {count} reviewed images ({epochs} epochs)...")
    model = YOLO(base_model)
    device = "0" if __import__("torch").cuda.is_available() else "cpu"
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=1280,
        batch=4,
        device=device,
        project=str(tmp_dir),
        name="finetune",
        exist_ok=True,
        verbose=False,
        freeze=22,
        lr0=0.001,
        lrf=0.1,
        mosaic=0.0,
        close_mosaic=0,
        augment=False,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        scale=0.2,
    )

    candidates = [
        tmp_dir / "finetune" / "weights" / "best.pt",
        tmp_dir / "finetune" / "weights" / "last.pt",
    ]
    for path in candidates:
        if path.exists():
            print(f"Updated model: {path}")
            return str(path)

    print("Quick fine-tune finished, but no updated weights were found. Keeping current model.")
    return base_model


def re_predict_unreviewed(
    images_dir: Path,
    labels_dir: Path,
    model_path: str,
    reviewed_indices: set,
    image_files: list,
    conf: float = 0.3,
) -> None:
    from ultralytics import YOLO

    model = YOLO(model_path)
    unreviewed = [i for i in range(len(image_files)) if i not in reviewed_indices]
    print(f"\nRe-predicting {len(unreviewed)} unreviewed images...")

    for idx in unreviewed:
        img_path = image_files[idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model(frame, conf=conf, verbose=False)
        r = results[0]
        lines = []
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                cls = int(r.boxes.cls[j])
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Re-prediction complete.")


def review_labels(
    images_dir: Path,
    labels_dir: Path,
    model_path: str,
    box_thickness: int,
    selected_thickness: int,
) -> None:
    image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not image_files:
        print("No images found.")
        return

    win_name = "Label Review - SPACE next | BACK prev | D delete | C class | N/P new | T retrain | Q quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    idx = 0
    selected = -1
    reviewed_indices = set()
    current_model = model_path

    while idx < len(image_files):
        img_path = image_files[idx]
        label_path = labels_dir / f"{img_path.stem}.txt"
        frame = cv2.imread(str(img_path))
        if frame is None:
            idx += 1
            continue

        h, w = frame.shape[:2]
        labels = load_labels(label_path)

        def refresh() -> None:
            display = draw_labels(
                frame,
                labels,
                selected,
                box_thickness=box_thickness,
                selected_thickness=selected_thickness,
            )
            info = f"[{idx + 1}/{len(image_files)}] {img_path.name} | {len(labels)} obj | reviewed={len(reviewed_indices)}"
            cv2.putText(display, info, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
            help_text = "SPACE next  BACK prev  D delete  C class  N vat  P person  T retrain  S select-next  Q quit"
            cv2.putText(display, help_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)

            scale = min(1400 / w, 1000 / h, 1.0)
            if scale < 1.0:
                display = cv2.resize(display, None, fx=scale, fy=scale)
            cv2.imshow(win_name, display)

        refresh()

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                save_labels(label_path, labels)
                reviewed_indices.add(idx)
                cv2.destroyAllWindows()
                print(f"Saved and quit at image {idx + 1}/{len(image_files)}.")
                return

            if key in (32, 13):
                save_labels(label_path, labels)
                reviewed_indices.add(idx)
                selected = -1
                idx += 1
                break

            if key == 8:
                save_labels(label_path, labels)
                selected = -1
                idx = max(0, idx - 1)
                break

            if key == ord("d") and 0 <= selected < len(labels):
                labels.pop(selected)
                selected = min(selected, len(labels) - 1)
                refresh()
                continue

            if key == ord("c") and 0 <= selected < len(labels):
                cls, cx, cy, bw, bh = labels[selected]
                cls = (cls + 1) % len(CLASS_NAMES)
                labels[selected] = (cls, cx, cy, bw, bh)
                refresh()
                continue

            if key == ord("s") and labels:
                selected = (selected + 1) % len(labels)
                refresh()
                continue

            if ord("0") <= key <= ord("9"):
                num = key - ord("0")
                if num < len(labels):
                    selected = num
                    refresh()
                continue

            if key == ord("t"):
                save_labels(label_path, labels)
                reviewed_indices.add(idx)
                cv2.destroyAllWindows()
                current_model = quick_finetune(images_dir, labels_dir, current_model, reviewed_indices, image_files)
                re_predict_unreviewed(images_dir, labels_dir, current_model, reviewed_indices, image_files, conf=0.3)
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                idx += 1
                break

            if key in (ord("n"), ord("p")):
                new_cls = 1 if key == ord("n") else 0
                color = CLASS_COLORS.get(new_cls, (255, 255, 255))
                draw_state = {"start": None, "end": None, "done": False}
                scale = min(1400 / w, 1000 / h, 1.0)

                def mouse_draw(event, x, y, flags, param) -> None:
                    sx = int(x / scale) if scale < 1.0 else x
                    sy = int(y / scale) if scale < 1.0 else y

                    if event == cv2.EVENT_LBUTTONDOWN:
                        if draw_state["start"] is None:
                            draw_state["start"] = (sx, sy)
                        else:
                            draw_state["end"] = (sx, sy)
                            draw_state["done"] = True

                    elif event == cv2.EVENT_MOUSEMOVE and draw_state["start"] is not None:
                        display = draw_labels(
                            frame,
                            labels,
                            selected,
                            box_thickness=box_thickness,
                            selected_thickness=selected_thickness,
                        )
                        sx0, sy0 = draw_state["start"]
                        cv2.rectangle(display, (sx0, sy0), (sx, sy), color, selected_thickness)
                        cv2.putText(display, f"Drawing: {CLASS_NAMES[new_cls]}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
                        if scale < 1.0:
                            display = cv2.resize(display, None, fx=scale, fy=scale)
                        cv2.imshow(win_name, display)

                cv2.setMouseCallback(win_name, mouse_draw)
                while not draw_state["done"]:
                    cv2.waitKey(30)
                cv2.setMouseCallback(win_name, lambda *args: None)

                if draw_state["start"] and draw_state["end"]:
                    x1, y1 = draw_state["start"]
                    x2, y2 = draw_state["end"]
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw_n = abs(x2 - x1) / w
                    bh_n = abs(y2 - y1) / h
                    labels.append((new_cls, cx, cy, bw_n, bh_n))
                    selected = len(labels) - 1
                    refresh()

    cv2.destroyAllWindows()
    print("Review complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Article-pack auto-label + review workflow.")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--labels", required=True, help="Label directory")
    parser.add_argument("--model", default="yolo11x.pt", help="Base model path")
    parser.add_argument("--conf", type=float, default=0.15, help="Auto-label confidence")
    parser.add_argument("--review", action="store_true", help="Skip initial auto-labeling and only review")
    parser.add_argument("--no-vat-candidates", action="store_true", help="Disable COCO-to-vat fallback mapping")
    parser.add_argument("--box-thickness", type=int, default=4, help="Regular bbox thickness in review window")
    parser.add_argument("--selected-thickness", type=int, default=7, help="Selected bbox thickness in review window")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Model: {args.model}")

    if not args.review:
        auto_label_all(
            images_dir,
            labels_dir,
            args.model,
            conf=args.conf,
            include_vat_candidates=not args.no_vat_candidates,
        )

    print("\nStarting review window...")
    print("Tip: review about 20-30 frames first, then press T for quick fine-tune.")
    review_labels(
        images_dir,
        labels_dir,
        args.model,
        box_thickness=args.box_thickness,
        selected_thickness=args.selected_thickness,
    )


if __name__ == "__main__":
    main()

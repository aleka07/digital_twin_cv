import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {0: "person", 1: "vat"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 165, 0)}
SELECTED_COLOR = (0, 0, 255)
DISPLAY_MAX_W = 1750
DISPLAY_MAX_H = 1100


@dataclass
class TrackBox:
    track_id: int
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


def box_area(box: TrackBox) -> int:
    return max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)


def box_iou(a: TrackBox, b: TrackBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_distance(a: TrackBox, b: TrackBox) -> float:
    acx = (a.x1 + a.x2) / 2
    acy = (a.y1 + a.y2) / 2
    bcx = (b.x1 + b.x2) / 2
    bcy = (b.y1 + b.y2) / 2
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def boxes_are_near(a: TrackBox, b: TrackBox) -> bool:
    if a.class_id != b.class_id:
        return False
    return box_iou(a, b) >= 0.15 or center_distance(a, b) <= 120


def suggest_prev_ids(current_boxes: list[TrackBox], prev_boxes: list[TrackBox]) -> dict[int, int]:
    suggestions: dict[int, int] = {}
    for idx, cur in enumerate(current_boxes):
        best_score = -1.0
        best_prev: TrackBox | None = None
        for prev in prev_boxes:
            if prev.class_id != cur.class_id:
                continue
            iou = box_iou(cur, prev)
            dist = center_distance(cur, prev)
            size_ref = max(40.0, ((cur.x2 - cur.x1) + (cur.y2 - cur.y1)) / 2)
            closeness = max(0.0, 1.0 - dist / (size_ref * 1.5))
            score = max(iou, closeness)
            if score > best_score:
                best_score = score
                best_prev = prev

        if best_prev is None:
            continue

        if best_prev.track_id == cur.track_id:
            continue

        if boxes_are_near(cur, best_prev):
            suggestions[idx] = best_prev.track_id
    return suggestions


def find_missing_prev_boxes(current_boxes: list[TrackBox], prev_boxes: list[TrackBox]) -> list[TrackBox]:
    missing: list[TrackBox] = []
    for prev in prev_boxes:
        matched = False
        for cur in current_boxes:
            if boxes_are_near(cur, prev):
                matched = True
                break
        if not matched:
            missing.append(prev)
    return missing


def has_real_annotations(path: Path) -> bool:
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            return True
    return False


def parse_track_file(path: Path) -> dict[int, list[TrackBox]]:
    by_frame: dict[int, list[TrackBox]] = {}
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
        by_frame.setdefault(frame_id, []).append(TrackBox(track_id, class_id, x1, y1, x2, y2))
    return by_frame


def save_track_file(path: Path, frames_data: dict[int, list[TrackBox]]) -> None:
    lines = ["# frame_id,track_id,class_id,x1,y1,x2,y2"]
    for frame_id in sorted(frames_data):
        frame_boxes = sorted(frames_data[frame_id], key=lambda box: (box.track_id, box.class_id, box.y1, box.x1))
        for box in frame_boxes:
            lines.append(
                f"{frame_id},{box.track_id},{box.class_id},{box.x1},{box.y1},{box.x2},{box.y2}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clamp_box(box: TrackBox, width: int, height: int) -> TrackBox:
    x1 = max(0, min(box.x1, width - 1))
    y1 = max(0, min(box.y1, height - 1))
    x2 = max(0, min(box.x2, width - 1))
    y2 = max(0, min(box.y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return TrackBox(box.track_id, box.class_id, x1, y1, x2, y2)


def get_display_scale(width: int, height: int) -> float:
    return min(DISPLAY_MAX_W / width, DISPLAY_MAX_H / height, 1.0)


def overlay_prompt(display: np.ndarray, prompt: str, value: str) -> np.ndarray:
    result = display.copy()
    height, width = result.shape[:2]
    panel_w = min(width - 40, 760)
    panel_h = 110
    x1 = max(20, (width - panel_w) // 2)
    y1 = 80
    x2 = x1 + panel_w
    y2 = y1 + panel_h
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(result, prompt, (x1 + 18, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    shown = value if value else "_"
    cv2.putText(result, shown, (x1 + 18, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    hint = "digits input  BACK delete  ENTER/SPACE confirm  ESC cancel"
    cv2.putText(result, hint, (x1 + 18, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2, cv2.LINE_AA)
    return result


def draw_boxes(
    frame: np.ndarray,
    boxes: list[TrackBox],
    selected_idx: int,
    frame_idx: int,
    total_frames: int,
    clip_name: str,
    wall_clock: str,
    suggestions: dict[int, int] | None = None,
    missing_prev_boxes: list[TrackBox] | None = None,
) -> np.ndarray:
    display = frame.copy()
    height, width = frame.shape[:2]
    suggestions = suggestions or {}
    missing_prev_boxes = missing_prev_boxes or []

    # Slightly darken the background so annotations stand out more.
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    display = cv2.addWeighted(overlay, 0.10, display, 0.90, 0)

    for idx, box in enumerate(boxes):
        color = SELECTED_COLOR if idx == selected_idx else CLASS_COLORS.get(box.class_id, (200, 200, 200))
        thickness = 7 if idx == selected_idx else 4
        cv2.rectangle(display, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
        label = f"[{idx}] id={box.track_id} {CLASS_NAMES.get(box.class_id, f'cls{box.class_id}')}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx = box.x1
        ty = max(35, box.y1 - 10)
        cv2.rectangle(display, (tx - 4, ty - th - 8), (tx + tw + 6, ty + 4), (0, 0, 0), -1)
        cv2.putText(display, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        if idx in suggestions:
            hint = f"prev id {suggestions[idx]} ?  [R]"
            (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            hy = min(height - 10, box.y2 + hh + 12)
            cv2.rectangle(display, (box.x1 - 4, hy - hh - 8), (box.x1 + hw + 6, hy + 4), (0, 0, 0), -1)
            cv2.putText(
                display,
                hint,
                (box.x1, hy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    top = f"{clip_name} [{frame_idx}/{total_frames}] {wall_clock} | {len(boxes)} obj"
    bottom = "SPACE next  BACK prev  S select  D delete  I id  C class  N/P new  M move  R use-prev-id  A add-missing  Q quit"
    cv2.rectangle(display, (6, 6), (min(width - 6, 1180), 58), (0, 0, 0), -1)
    cv2.rectangle(display, (6, height - 42), (min(width - 6, 1240), height - 6), (0, 0, 0), -1)
    cv2.putText(display, top, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        display,
        bottom,
        (14, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    if 0 <= selected_idx < len(boxes):
        selected = boxes[selected_idx]
        sx1, sy1, sx2, sy2 = selected.x1, selected.y1, selected.x2, selected.y2
        pad = 30
        cx1 = max(0, sx1 - pad)
        cy1 = max(0, sy1 - pad)
        cx2 = min(width, sx2 + pad)
        cy2 = min(height, sy2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            zoom_w = 320
            zoom_h = 240
            zoom = cv2.resize(crop, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
            box_w = max(1, cx2 - cx1)
            box_h = max(1, cy2 - cy1)
            zx1 = int((sx1 - cx1) / box_w * zoom_w)
            zy1 = int((sy1 - cy1) / box_h * zoom_h)
            zx2 = int((sx2 - cx1) / box_w * zoom_w)
            zy2 = int((sy2 - cy1) / box_h * zoom_h)
            cv2.rectangle(zoom, (zx1, zy1), (zx2, zy2), SELECTED_COLOR, 4)

            px1 = max(10, width - zoom_w - 20)
            py1 = 70
            px2 = px1 + zoom_w
            py2 = py1 + zoom_h
            display[py1:py2, px1:px2] = zoom
            cv2.rectangle(display, (px1, py1), (px2, py2), (255, 255, 255), 2)
            info = f"sel id={selected.track_id} cls={CLASS_NAMES.get(selected.class_id, selected.class_id)}"
            if selected_idx in suggestions:
                info += f"  prev->{suggestions[selected_idx]}"
            cv2.rectangle(display, (px1, py2 + 4), (px2, py2 + 34), (0, 0, 0), -1)
            cv2.putText(display, info, (px1 + 8, py2 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    if missing_prev_boxes:
        panel_lines = ["missing from prev [A]:"]
        for box in missing_prev_boxes[:6]:
            panel_lines.append(f"id={box.track_id} {CLASS_NAMES.get(box.class_id, box.class_id)}")
        if len(missing_prev_boxes) > 6:
            panel_lines.append(f"+{len(missing_prev_boxes) - 6} more")

        panel_w = min(360, width - 20)
        panel_h = 34 + 28 * len(panel_lines)
        px1 = max(10, width - panel_w - 20)
        py1 = height - panel_h - 60
        px2 = px1 + panel_w
        py2 = py1 + panel_h
        cv2.rectangle(display, (px1, py1), (px2, py2), (0, 0, 0), -1)
        cv2.rectangle(display, (px1, py1), (px2, py2), (0, 255, 255), 2)
        for line_idx, line in enumerate(panel_lines):
            color = (0, 255, 255) if line_idx == 0 else (255, 255, 255)
            cv2.putText(
                display,
                line,
                (px1 + 10, py1 + 28 + line_idx * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    scale = get_display_scale(width, height)
    if scale < 1.0:
        display = cv2.resize(display, None, fx=scale, fy=scale)
    return display


def ask_int_in_window(
    win_name: str,
    render_fn,
    prompt: str,
    default: int | None = None,
) -> int | None:
    value = "" if default is None else str(default)
    while True:
        display = overlay_prompt(render_fn(), prompt, value)
        cv2.imshow(win_name, display)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            return None
        if key in (13, 32):
            if value:
                return int(value)
            return default
        if key in (8, 127):
            value = value[:-1]
            continue
        if ord("0") <= key <= ord("9"):
            value += chr(key)


def draw_new_box(
    win_name: str,
    frame: np.ndarray,
    existing: list[TrackBox],
    class_id: int,
    selected_idx: int,
) -> tuple[int, int, int, int] | None:
    draw_state = {"start": None, "end": None, "done": False, "cancel": False}
    height, width = frame.shape[:2]
    scale = get_display_scale(width, height)

    def redraw(cursor: tuple[int, int] | None = None) -> None:
        display = draw_boxes(frame, existing, selected_idx, 0, 0, "draw", "")
        if draw_state["start"] is not None and cursor is not None:
            x1, y1 = draw_state["start"]
            x2, y2 = cursor
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            sx1 = int(round(x1 * scale))
            sy1 = int(round(y1 * scale))
            sx2 = int(round(x2 * scale))
            sy2 = int(round(y2 * scale))
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), color, 6)
        cv2.imshow(win_name, display)

    def mouse_cb(event: int, x: int, y: int, flags: int, param: object) -> None:
        px = int(x / scale) if scale < 1.0 else x
        py = int(y / scale) if scale < 1.0 else y
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            if draw_state["start"] is None:
                draw_state["start"] = (px, py)
            else:
                draw_state["end"] = (px, py)
                draw_state["done"] = True
        elif event == cv2.EVENT_MOUSEMOVE and draw_state["start"] is not None:
            redraw((px, py))

    redraw()
    cv2.setMouseCallback(win_name, mouse_cb)

    while not draw_state["done"] and not draw_state["cancel"]:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            draw_state["cancel"] = True
            break

    cv2.setMouseCallback(win_name, lambda *args: None)

    if draw_state["cancel"] or draw_state["start"] is None or draw_state["end"] is None:
        return None

    x1, y1 = draw_state["start"]
    x2, y2 = draw_state["end"]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def move_selected_box(
    win_name: str,
    frame: np.ndarray,
    boxes: list[TrackBox],
    selected_idx: int,
) -> TrackBox | None:
    if not (0 <= selected_idx < len(boxes)):
        return None

    current = boxes[selected_idx]
    updated = TrackBox(current.track_id, current.class_id, current.x1, current.y1, current.x2, current.y2)
    height, width = frame.shape[:2]
    scale = get_display_scale(width, height)
    drag_state = {"dragging": False, "offset_x": 0, "offset_y": 0, "done": False, "cancel": False}

    def redraw() -> None:
        temp_boxes = boxes.copy()
        temp_boxes[selected_idx] = updated
        display = draw_boxes(frame, temp_boxes, selected_idx, 0, 0, "move", "")
        cv2.imshow(win_name, display)

    def mouse_cb(event: int, x: int, y: int, flags: int, param: object) -> None:
        px = int(x / scale) if scale < 1.0 else x
        py = int(y / scale) if scale < 1.0 else y
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            if updated.x1 <= px <= updated.x2 and updated.y1 <= py <= updated.y2:
                drag_state["dragging"] = True
                drag_state["offset_x"] = px - updated.x1
                drag_state["offset_y"] = py - updated.y1
        elif event == cv2.EVENT_MOUSEMOVE and drag_state["dragging"]:
            box_w = updated.x2 - updated.x1
            box_h = updated.y2 - updated.y1
            new_x1 = px - drag_state["offset_x"]
            new_y1 = py - drag_state["offset_y"]
            new_x2 = new_x1 + box_w
            new_y2 = new_y1 + box_h
            updated.x1 = max(0, min(new_x1, width - 1))
            updated.y1 = max(0, min(new_y1, height - 1))
            updated.x2 = max(0, min(new_x2, width - 1))
            updated.y2 = max(0, min(new_y2, height - 1))
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and drag_state["dragging"]:
            drag_state["dragging"] = False

    redraw()
    cv2.setMouseCallback(win_name, mouse_cb)

    while not drag_state["done"] and not drag_state["cancel"]:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            drag_state["done"] = True
        elif key in (27, ord("q")):
            drag_state["cancel"] = True

    cv2.setMouseCallback(win_name, lambda *args: None)
    if drag_state["cancel"]:
        return None
    return clamp_box(updated, width, height)


def review_clip(
    clip_dir: Path,
    source_name: str,
    gt_name: str,
) -> None:
    frames_dir = clip_dir / "frames"
    manifest_path = clip_dir / "frame_manifest.csv"
    source_path = clip_dir / source_name
    gt_path = clip_dir / gt_name

    manifest_rows = read_manifest(manifest_path)
    if not manifest_rows:
        print(f"No frame manifest rows found in {manifest_path}")
        return

    active_source = gt_path if has_real_annotations(gt_path) else source_path
    frame_data = parse_track_file(active_source)
    print(f"Starting from: {active_source}")
    for row in manifest_rows:
        frame_id = int(row["frame_id"])
        frame_data.setdefault(frame_id, [])

    total_boxes = sum(len(items) for items in frame_data.values())
    print(f"Loaded {total_boxes} boxes across {len(frame_data)} frames.")

    win_name = "Tracking Review - SPACE next | BACK prev | D delete | I id | C class | N/P new | M move | Q quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    idx = 0
    selected_idx = -1
    clip_name = clip_dir.name

    while 0 <= idx < len(manifest_rows):
        row = manifest_rows[idx]
        frame_id = int(row["frame_id"])
        frame_path = frames_dir / row["image_name"]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            idx += 1
            continue

        height, width = frame.shape[:2]
        boxes = [clamp_box(box, width, height) for box in frame_data.get(frame_id, [])]
        frame_data[frame_id] = boxes
        prev_boxes = frame_data.get(frame_id - 1, [])
        suggestions = suggest_prev_ids(boxes, prev_boxes)
        missing_prev_boxes = find_missing_prev_boxes(boxes, prev_boxes)

        def render_base() -> np.ndarray:
            return draw_boxes(
                frame,
                boxes,
                selected_idx,
                frame_id,
                len(manifest_rows),
                clip_name,
                row["wall_clock"],
                suggestions=suggestions,
                missing_prev_boxes=missing_prev_boxes,
            )

        def refresh() -> None:
            cv2.imshow(win_name, render_base())

        refresh()

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                save_track_file(gt_path, frame_data)
                cv2.destroyAllWindows()
                print(f"Saved and quit: {gt_path}")
                return

            if key in (32, 13):
                cv2.waitKey(1)
                save_track_file(gt_path, frame_data)
                selected_idx = -1
                idx += 1
                break

            if key == 8:
                cv2.waitKey(1)
                save_track_file(gt_path, frame_data)
                selected_idx = -1
                idx = max(0, idx - 1)
                break

            if key == ord("s") and boxes:
                selected_idx = (selected_idx + 1) % len(boxes)
                refresh()
                continue

            if ord("0") <= key <= ord("9"):
                num = key - ord("0")
                if num < len(boxes):
                    selected_idx = num
                    refresh()
                continue

            if key == ord("d") and 0 <= selected_idx < len(boxes):
                boxes.pop(selected_idx)
                selected_idx = min(selected_idx, len(boxes) - 1)
                refresh()
                continue

            if key == ord("c") and 0 <= selected_idx < len(boxes):
                box = boxes[selected_idx]
                box.class_id = (box.class_id + 1) % len(CLASS_NAMES)
                refresh()
                continue

            if key == ord("i") and 0 <= selected_idx < len(boxes):
                box = boxes[selected_idx]
                new_track_id = ask_int_in_window(win_name, render_base, "New track_id", box.track_id)
                if new_track_id is not None:
                    box.track_id = new_track_id
                    suggestions = suggest_prev_ids(boxes, prev_boxes)
                    missing_prev_boxes = find_missing_prev_boxes(boxes, prev_boxes)
                    refresh()
                continue

            if key == ord("r") and 0 <= selected_idx < len(boxes):
                if selected_idx in suggestions:
                    boxes[selected_idx].track_id = suggestions[selected_idx]
                    suggestions = suggest_prev_ids(boxes, prev_boxes)
                    missing_prev_boxes = find_missing_prev_boxes(boxes, prev_boxes)
                    refresh()
                continue

            if key == ord("a") and missing_prev_boxes:
                add_box = missing_prev_boxes[0]
                boxes.append(TrackBox(add_box.track_id, add_box.class_id, add_box.x1, add_box.y1, add_box.x2, add_box.y2))
                selected_idx = len(boxes) - 1
                suggestions = suggest_prev_ids(boxes, prev_boxes)
                missing_prev_boxes = find_missing_prev_boxes(boxes, prev_boxes)
                refresh()
                continue

            if key == ord("m") and 0 <= selected_idx < len(boxes):
                moved = move_selected_box(win_name, frame, boxes, selected_idx)
                if moved is not None:
                    boxes[selected_idx] = moved
                    suggestions = suggest_prev_ids(boxes, prev_boxes)
                    missing_prev_boxes = find_missing_prev_boxes(boxes, prev_boxes)
                    refresh()
                else:
                    refresh()
                continue

            if key in (ord("n"), ord("p")):
                class_id = 1 if key == ord("n") else 0
                suggested = 1
                if boxes:
                    suggested = max(box.track_id for box in boxes) + 1
                prompt = f"track_id for new {CLASS_NAMES[class_id]}"
                track_id = ask_int_in_window(win_name, render_base, prompt, suggested)
                if track_id is None:
                    refresh()
                    continue
                coords = draw_new_box(win_name, frame, boxes, class_id, selected_idx)
                if coords is None:
                    refresh()
                    continue
                x1, y1, x2, y2 = coords
                boxes.append(TrackBox(track_id, class_id, x1, y1, x2, y2))
                selected_idx = len(boxes) - 1
                suggestions = suggest_prev_ids(boxes, prev_boxes)
                missing_prev_boxes = find_missing_prev_boxes(boxes, prev_boxes)
                refresh()
                continue

    save_track_file(gt_path, frame_data)
    cv2.destroyAllWindows()
    print(f"Saved clip: {gt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive tracking GT review tool for article_pack clips.")
    parser.add_argument(
        "--clip-dir",
        required=True,
        help="Clip directory, e.g. article_pack/tracking_benchmark/clip_A",
    )
    parser.add_argument(
        "--source",
        default="draft_bytetrack.txt",
        help="Starting annotation source inside the clip dir",
    )
    parser.add_argument(
        "--gt",
        default="gt_tracks.txt",
        help="Ground truth file inside the clip dir",
    )
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    review_clip(clip_dir, args.source, args.gt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted from console. Progress up to the last saved frame is kept.")

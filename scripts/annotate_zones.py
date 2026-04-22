import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


DISPLAY_MAX_W = 1750
DISPLAY_MAX_H = 1100
POINT_RADIUS = 7
PICK_RADIUS = 18


def ensure_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def get_display_scale(width: int, height: int) -> float:
    return min(DISPLAY_MAX_W / width, DISPLAY_MAX_H / height, 1.0)


def load_zones(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_zones(path: Path, zones: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(zones, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def auto_color(index: int) -> list[int]:
    palette = [
        [60, 180, 255],
        [80, 220, 120],
        [255, 120, 60],
        [180, 80, 255],
        [255, 220, 80],
        [80, 255, 220],
        [220, 120, 255],
    ]
    return palette[index % len(palette)]


def render_scene(
    image: np.ndarray,
    zones: list[dict],
    selected_idx: int,
    draft_points: list[tuple[int, int]] | None = None,
    draft_name: str | None = None,
    hover_point: tuple[int, int] | None = None,
) -> np.ndarray:
    display = image.copy()
    height, width = image.shape[:2]

    for idx, zone in enumerate(zones):
        pts = np.array(zone["points"], dtype=np.int32)
        color = tuple(int(c) for c in zone.get("color", auto_color(idx)))
        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], color)
        alpha = 0.16 if idx != selected_idx else 0.28
        display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)
        thickness = 5 if idx == selected_idx else 3
        cv2.polylines(display, [pts], True, color, thickness, cv2.LINE_AA)

        for p_idx, (px, py) in enumerate(zone["points"]):
            point_color = (0, 0, 255) if idx == selected_idx else color
            cv2.circle(display, (px, py), POINT_RADIUS, point_color, -1, cv2.LINE_AA)
            if idx == selected_idx:
                cv2.putText(
                    display,
                    str(p_idx),
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        label = zone["name"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(display, (cx - tw // 2 - 6, cy - th - 8), (cx + tw // 2 + 6, cy + 8), (0, 0, 0), -1)
        cv2.putText(display, label, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    if draft_points:
        draft_color = (0, 255, 255)
        for px, py in draft_points:
            cv2.circle(display, (px, py), POINT_RADIUS, draft_color, -1, cv2.LINE_AA)
        if len(draft_points) >= 2:
            cv2.polylines(display, [np.array(draft_points, dtype=np.int32)], False, draft_color, 3, cv2.LINE_AA)
        if hover_point is not None and draft_points:
            cv2.line(display, draft_points[-1], hover_point, draft_color, 2, cv2.LINE_AA)
        if draft_name:
            label = f"new: {draft_name}"
            cv2.rectangle(display, (20, 70), (340, 108), (0, 0, 0), -1)
            cv2.putText(display, label, (28, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.8, draft_color, 2, cv2.LINE_AA)

    top = f"zones={len(zones)}"
    if 0 <= selected_idx < len(zones):
        top += f" | selected={zones[selected_idx]['name']}"
    bottom = "N new  [/] select  drag point  ENTER close-zone  X del-point  D del-zone  R rename  S save  P preview  Q quit"
    cv2.rectangle(display, (8, 8), (min(width - 8, 1100), 56), (0, 0, 0), -1)
    cv2.rectangle(display, (8, height - 40), (min(width - 8, 1450), height - 8), (0, 0, 0), -1)
    cv2.putText(display, top, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(display, bottom, (18, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2, cv2.LINE_AA)

    scale = get_display_scale(width, height)
    if scale < 1.0:
        display = cv2.resize(display, None, fx=scale, fy=scale)
    return display


def overlay_prompt(display: np.ndarray, prompt: str, value: str) -> np.ndarray:
    result = display.copy()
    height, width = result.shape[:2]
    panel_w = min(width - 40, 860)
    panel_h = 120
    x1 = max(20, (width - panel_w) // 2)
    y1 = 70
    x2 = x1 + panel_w
    y2 = y1 + panel_h
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(result, prompt, (x1 + 18, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    shown = value if value else "_"
    cv2.putText(result, shown, (x1 + 18, y1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    hint = "letters/digits/_/-  BACK delete  ENTER/SPACE confirm  ESC cancel"
    cv2.putText(result, hint, (x1 + 18, y2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2, cv2.LINE_AA)
    return result


def ask_text_in_window(win_name: str, render_fn, prompt: str, default: str = "") -> str | None:
    value = default
    while True:
        cv2.imshow(win_name, overlay_prompt(render_fn(), prompt, value))
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            return None
        if key in (13, 32):
            return value.strip() or default
        if key in (8, 127):
            value = value[:-1]
            continue
        ch = chr(key)
        if ch.isalnum() or ch in "_-":
            value += ch


def find_nearest_point(points: list[list[int]] | list[tuple[int, int]], x: int, y: int) -> int | None:
    best_idx = None
    best_dist = None
    for idx, (px, py) in enumerate(points):
        dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if dist <= PICK_RADIUS and (best_dist is None or dist < best_dist):
            best_idx = idx
            best_dist = dist
    return best_idx


def export_preview(image: np.ndarray, zones: list[dict], path: Path) -> None:
    preview = render_scene(image, zones, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def annotate(image_path: Path, zones_path: Path, output_path: Path, preview_path: Path | None, render_only: bool) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    zones = load_zones(zones_path)
    if output_path.exists() and output_path != zones_path:
        zones = load_zones(output_path)

    if render_only:
        export_preview(image, zones, preview_path or output_path.with_suffix(".preview.jpg"))
        return

    win_name = "Zone Annotator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    height, width = image.shape[:2]
    scale = get_display_scale(width, height)

    selected_idx = 0 if zones else -1
    draft_points: list[tuple[int, int]] = []
    draft_name: str | None = None
    hover_point: tuple[int, int] | None = None
    drag_state = {"active": False, "point_idx": None}

    def render_base() -> np.ndarray:
        return render_scene(image, zones, selected_idx, draft_points or None, draft_name, hover_point)

    def refresh() -> None:
        cv2.imshow(win_name, render_base())

    def mouse_cb(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal hover_point
        px = int(x / scale) if scale < 1.0 else x
        py = int(y / scale) if scale < 1.0 else y
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        hover_point = (px, py)

        if draft_name is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                draft_points.append((px, py))
                refresh()
            elif event == cv2.EVENT_MOUSEMOVE:
                refresh()
            return

        if not (0 <= selected_idx < len(zones)):
            return

        points = zones[selected_idx]["points"]
        if event == cv2.EVENT_LBUTTONDOWN:
            point_idx = find_nearest_point(points, px, py)
            if point_idx is not None:
                drag_state["active"] = True
                drag_state["point_idx"] = point_idx
        elif event == cv2.EVENT_MOUSEMOVE and drag_state["active"]:
            point_idx = drag_state["point_idx"]
            if point_idx is not None:
                points[point_idx] = [px, py]
                refresh()
        elif event == cv2.EVENT_LBUTTONUP:
            drag_state["active"] = False
            drag_state["point_idx"] = None

    cv2.setMouseCallback(win_name, mouse_cb)
    refresh()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            save_zones(output_path, zones)
            if preview_path:
                export_preview(image, zones, preview_path)
            cv2.destroyAllWindows()
            print(f"Saved and quit: {output_path}")
            return

        if key == ord("s"):
            save_zones(output_path, zones)
            if preview_path:
                export_preview(image, zones, preview_path)
            print(f"Saved: {output_path}")
            refresh()
            continue

        if key == ord("p"):
            out = preview_path or output_path.with_suffix(".preview.jpg")
            export_preview(image, zones, out)
            print(f"Preview saved: {out}")
            refresh()
            continue

        if key == ord("n"):
            default_name = f"zone_{len(zones) + 1}"
            name = ask_text_in_window(win_name, render_base, "New zone name", default_name)
            if name:
                draft_name = name
                draft_points = []
                hover_point = None
            refresh()
            continue

        if key in (13, 32) and draft_name is not None:
            if len(draft_points) >= 3:
                zones.append(
                    {
                        "name": draft_name,
                        "points": [[int(x), int(y)] for x, y in draft_points],
                        "color": auto_color(len(zones)),
                    }
                )
                selected_idx = len(zones) - 1
            draft_name = None
            draft_points = []
            hover_point = None
            refresh()
            continue

        if key == ord("[") and zones:
            selected_idx = (selected_idx - 1) % len(zones)
            refresh()
            continue

        if key == ord("]") and zones:
            selected_idx = (selected_idx + 1) % len(zones)
            refresh()
            continue

        if key == ord("d") and 0 <= selected_idx < len(zones):
            zones.pop(selected_idx)
            selected_idx = min(selected_idx, len(zones) - 1)
            refresh()
            continue

        if key == ord("x") and 0 <= selected_idx < len(zones):
            points = zones[selected_idx]["points"]
            if hover_point is not None:
                point_idx = find_nearest_point(points, hover_point[0], hover_point[1])
                if point_idx is not None and len(points) > 3:
                    points.pop(point_idx)
            refresh()
            continue

        if key == ord("r") and 0 <= selected_idx < len(zones):
            current = zones[selected_idx]["name"]
            renamed = ask_text_in_window(win_name, render_base, "Rename zone", current)
            if renamed:
                zones[selected_idx]["name"] = renamed
            refresh()
            continue

        if key == ord("c") and 0 <= selected_idx < len(zones):
            zones[selected_idx]["color"] = auto_color(selected_idx + 1)
            refresh()
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive polygon zone annotation for article_pack.")
    parser.add_argument(
        "--image",
        default="article_pack/tracking_benchmark/clip_B/frames/clip_B_000001.jpg",
        help="Background image for zone drawing",
    )
    parser.add_argument(
        "--zones",
        default="article_pack/zones/cam10_zone_suggestions.json",
        help="Input zones JSON",
    )
    parser.add_argument(
        "--output",
        default="article_pack/zones/cam10_zones_final.json",
        help="Output zones JSON",
    )
    parser.add_argument(
        "--preview-out",
        default="article_pack/zones/cam10_zone_suggestions_preview.jpg",
        help="Optional preview image output",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render zones to preview image and exit",
    )
    args = parser.parse_args()

    ensure_project_root()
    annotate(
        image_path=Path(args.image),
        zones_path=Path(args.zones),
        output_path=Path(args.output),
        preview_path=Path(args.preview_out) if args.preview_out else None,
        render_only=args.render_only,
    )


if __name__ == "__main__":
    main()

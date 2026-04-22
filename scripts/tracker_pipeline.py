from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class Detection:
    track_id: int
    cls: int
    cls_name: str
    conf: float
    bbox: tuple

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


_TRACK_COLORS = [
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
    (128, 0, 255),
    (0, 255, 128),
    (200, 200, 0),
    (200, 0, 200),
    (0, 200, 200),
    (100, 255, 100),
    (255, 100, 100),
    (100, 100, 255),
    (180, 255, 50),
    (50, 180, 255),
]


def _color_for_id(track_id: int) -> tuple:
    if track_id < 0:
        return (200, 200, 200)
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


class TrackerPipeline:
    TRACKER_CONFIGS = {
        "bytetrack": "bytetrack.yaml",
        "botsort": "botsort.yaml",
    }

    def __init__(
        self,
        model_path: str,
        tracker: str = "bytetrack",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 1280,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        tracker_path = Path(tracker)
        if tracker_path.suffix == ".yaml" or tracker_path.exists():
            self.tracker_yaml = str(tracker_path)
        else:
            self.tracker_yaml = self.TRACKER_CONFIGS.get(tracker, f"{tracker}.yaml")
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_names = self.model.names
        self._frame_count = 0
        self._all_track_ids = set()

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_yaml,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        r = results[0]
        self._frame_count += 1

        detections = []
        if r.boxes is not None and len(r.boxes) > 0:
            for j in range(len(r.boxes)):
                cls = int(r.boxes.cls[j])
                conf = float(r.boxes.conf[j])
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[j].cpu().numpy())
                track_id = -1
                if r.boxes.id is not None:
                    track_id = int(r.boxes.id[j])
                    self._all_track_ids.add(track_id)
                cls_name = self.class_names.get(cls, f"cls{cls}")
                detections.append(
                    Detection(
                        track_id=track_id,
                        cls=cls,
                        cls_name=cls_name,
                        conf=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_conf: bool = True,
        show_track_id: bool = True,
        thickness: int = 2,
    ) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = _color_for_id(det.track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            parts = []
            if show_track_id and det.track_id >= 0:
                parts.append(f"#{det.track_id}")
            parts.append(det.cls_name)
            if show_conf:
                parts.append(f"{det.conf:.2f}")
            label = " ".join(parts)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def unique_track_ids(self) -> set:
        return self._all_track_ids.copy()

    def reset(self) -> None:
        self.model.predictor = None
        self._frame_count = 0
        self._all_track_ids = set()

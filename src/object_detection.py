import cv2 # OpenCV is used here for drawing detection boxes and labels

__all__ = ["process_object_detection"]

_YOLO_MODEL_PATH = "yolov8n.pt" # Small pretrained YOLO model: fast enough for a beginner ADAS prototype
_CONFIDENCE_THRESHOLD = 0.35 # Ignore weak detections to reduce noisy boxes on road footage
_NEAR_BOX_HEIGHT = 220 # Large pixel height means the object is visually close to the camera
_MEDIUM_BOX_HEIGHT = 120 # Medium pixel height means the object may be relevant but is less urgent
_IN_PATH_OFFSET = 120 # Pixel distance from lane center that counts as directly in the driving path
_NEAR_PATH_OFFSET = 220 # Pixel distance from lane center that counts as close to the driving path
_TARGET_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}
_CLASS_COLORS = {
    "person": (0, 255, 255), # Yellow for pedestrians
    "car": (0, 255, 0), # Green for vehicles
    "motorcycle": (0, 255, 0),
    "bus": (0, 255, 0),
    "truck": (0, 255, 0),
}
_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
_WARNING_PRIORITY = {
    "CAUTION": 1,
    "PEDESTRIAN WARNING": 2,
    "COLLISION WARNING": 3,
}

_model = None


def _load_yolo_model():
    # Load the pretrained YOLO model only once, then reuse it for every video frame.
    global _model

    if _model is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Object detection requires the ultralytics package. "
                "Install it with: pip install ultralytics"
            ) from exc

        _model = YOLO(_YOLO_MODEL_PATH)

    return _model


def _draw_detection_label(image, label, x1, y1, color):
    # Draw label text on a filled rectangle so it stays readable over complex road scenes.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size

    label_y1 = max(y1 - text_height - baseline - 8, 0)
    label_y2 = y1
    label_x2 = x1 + text_width + 8

    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
    cv2.putText(
        image,
        label,
        (x1 + 4, y1 - baseline - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )


def _estimate_distance_level(box_height):
    # Treat bounding-box height as a rough distance proxy: larger boxes usually mean closer objects.
    if box_height >= _NEAR_BOX_HEIGHT:
        return "near"
    if box_height >= _MEDIUM_BOX_HEIGHT:
        return "medium"
    return "far"


def _estimate_path_status(object_center_x, lane_info):
    # Compare the object's horizontal center against the lane center from lane detection.
    if lane_info is None:
        return "unknown"

    lane_center_x = lane_info["lane_center_x"]
    center_offset = abs(object_center_x - lane_center_x)

    if center_offset <= _IN_PATH_OFFSET:
        return "in_path"
    if center_offset <= _NEAR_PATH_OFFSET:
        return "near_path"
    return "outside_path"


def _get_lane_center_offset(object_center_x, lane_info):
    # Temporary debug helper for printing how far a detected object is from the lane center.
    if lane_info is None:
        return None, None

    lane_center_x = lane_info["lane_center_x"]
    return lane_center_x, abs(object_center_x - lane_center_x)


def _get_collision_warning(class_name, distance_level, path_status):
    # Combine class, visual closeness, and lane-relative position into a first-pass warning rule.
    if path_status == "unknown":
        return None

    if class_name == "person" and distance_level in {"near", "medium"} and path_status == "in_path":
        return "PEDESTRIAN WARNING"

    if class_name in _VEHICLE_CLASSES:
        if distance_level == "near" and path_status == "in_path":
            return "COLLISION WARNING"
        if distance_level in {"near", "medium"} and path_status == "near_path":
            return "CAUTION"

    return None


def _draw_warning_banner(image, warning):
    # Draw the highest-level warning at the top of the frame so it is easy to notice.
    cv2.rectangle(image, (20, 20), (520, 75), (0, 0, 255), cv2.FILLED)
    cv2.putText(
        image,
        warning,
        (35, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
    )


def _choose_highest_priority_warning(current_warning, new_warning):
    # Keep the most serious warning if multiple objects are detected in the same frame.
    if new_warning is None:
        return current_warning
    if current_warning is None:
        return new_warning
    if _WARNING_PRIORITY[new_warning] > _WARNING_PRIORITY[current_warning]:
        return new_warning
    return current_warning


def process_object_detection(frame, lane_info=None):
    # Run pretrained YOLO detection on one frame and visualize vehicles and pedestrians.
    model = _load_yolo_model()
    detection_overlay = frame.copy()
    results = model(frame, verbose=False)
    active_warning = None

    if not results:
        return detection_overlay

    class_names = results[0].names
    boxes = results[0].boxes

    if boxes is None:
        return detection_overlay

    for box in boxes:
        confidence = float(box.conf[0])
        if confidence < _CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        if class_name not in _TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_height = y2 - y1
        object_center_x = (x1 + x2) / 2
        distance_level = _estimate_distance_level(box_height)
        path_status = _estimate_path_status(object_center_x, lane_info)
        warning = _get_collision_warning(class_name, distance_level, path_status)
        color = _CLASS_COLORS[class_name]
        label = f"{class_name} {confidence:.2f} {distance_level}"

        cv2.rectangle(detection_overlay, (x1, y1), (x2, y2), color, 2)
        _draw_detection_label(detection_overlay, label, x1, y1, color)

        if warning is not None:
            lane_center_x, absolute_offset = _get_lane_center_offset(object_center_x, lane_info)
            print(
                "Warning debug: "
                f"lane_center_x={lane_center_x}, "
                f"object_center_x={object_center_x}, "
                f"absolute_offset={absolute_offset}, "
                f"path_status={path_status}, "
                f"distance_level={distance_level}, "
                f"warning={warning}"
            )
            active_warning = _choose_highest_priority_warning(active_warning, warning)
            cv2.rectangle(detection_overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)

    if active_warning is not None:
        _draw_warning_banner(detection_overlay, active_warning)

    return detection_overlay

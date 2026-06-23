import cv2 # OpenCV is used here for drawing detection boxes and labels

__all__ = ["process_object_detection"]

_YOLO_MODEL_PATH = "yolov8n.pt" # Small pretrained YOLO model: fast enough for a beginner ADAS prototype
_CONFIDENCE_THRESHOLD = 0.35 # Ignore weak detections to reduce noisy boxes on road footage
_TARGET_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}
_CLASS_COLORS = {
    "person": (0, 255, 255), # Yellow for pedestrians
    "car": (0, 255, 0), # Green for vehicles
    "motorcycle": (0, 255, 0),
    "bus": (0, 255, 0),
    "truck": (0, 255, 0),
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


def process_object_detection(frame):
    # Run pretrained YOLO detection on one frame and visualize vehicles and pedestrians.
    model = _load_yolo_model()
    detection_overlay = frame.copy()
    results = model(frame, verbose=False)

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
        color = _CLASS_COLORS[class_name]
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(detection_overlay, (x1, y1), (x2, y2), color, 2)
        _draw_detection_label(detection_overlay, label, x1, y1, color)

    return detection_overlay

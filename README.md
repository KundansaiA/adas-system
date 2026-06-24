# ADAS Computer Vision System

An educational Advanced Driver Assistance System (ADAS) project built with Python and OpenCV. The goal is to process road footage, detect lane boundaries, and grow the system into a modular driver assistance pipeline with vehicle detection, pedestrian detection, and collision risk warnings.

This project is focused on computer vision understanding and production-style software organization. It is not a self-driving car project and does not control any vehicle hardware.

## Project Goals

- Build a resume-quality computer vision project around automotive safety.
- Learn the core OpenCV pipeline used in classical lane detection.
- Keep the code modular so lane detection, object detection, and warning logic can evolve independently.
- Document design decisions clearly enough to explain in interviews.

## Current Status

The current implementation focuses on lane-detection fundamentals:

- Loads a road video with OpenCV.
- Converts each frame to grayscale.
- Applies Canny edge detection to highlight strong image gradients.
- Uses a trapezoid-shaped region of interest to focus on the road area.
- Runs the Probabilistic Hough Transform to detect line segments.
- Filters lane candidates by slope and image position.
- Averages left-lane and right-lane candidates into representative lane lines.
- Displays debug windows for masked edges, raw Hough lines, filtered Hough lines, and averaged lanes.

Planned features include object detection, pedestrian detection, collision-risk analysis, and driver-assistance warnings.

## Why This Matters

ADAS systems combine perception, decision logic, and user-facing warnings. This project starts with the perception layer: turning raw video frames into useful road features. Lane detection is a good first milestone because it introduces many computer vision concepts used in larger systems:

- Image preprocessing
- Edge detection
- Region masking
- Geometric filtering
- Temporal frame-by-frame processing
- Debug visualization

## Tech Stack

- Python
- OpenCV for video processing, lane detection, and visualization
- NumPy for frame and geometry calculations
- Ultralytics YOLOv8 for pretrained vehicle and pedestrian detection
- Git and GitHub for version control and project hosting

## Project Structure

```text
adas-system/
├── data/
│   └── raw_videos/
│       └── road.mp4
├── docs/
│   ├── architecture.md
│   ├── project_scope.md
│   └── roadmap.md
├── outputs/
├── src/
│   ├── lane_detection.py
│   ├── main.py
│   ├── object_detection.py
│   └── utils.py
├── AGENTS.md
└── README.md
```

## How The Lane Detection Pipeline Works

1. **Video input**

   OpenCV reads the driving footage one frame at a time using `cv2.VideoCapture`.

2. **Grayscale conversion**

   Each color frame is converted to grayscale. Lane detection does not initially need color information, and grayscale simplifies edge detection.

3. **Canny edge detection**

   Canny detects sharp changes in pixel intensity. Lane markings usually create strong edges against the road surface.

4. **Region of interest masking**

   A trapezoid mask keeps only the part of the image where lane lines are expected to appear. This reduces noise from the sky, trees, buildings, and unrelated road features.

5. **Hough line detection**

   The Probabilistic Hough Transform finds straight line segments from the edge image.

6. **Lane candidate filtering**

   Detected line segments are filtered by slope and position:

   - Negative-slope lines on the left side are treated as left-lane candidates.
   - Positive-slope lines on the right side are treated as right-lane candidates.
   - Mostly horizontal or vertical lines are ignored.

7. **Lane averaging**

   Candidate segments are converted into slope-intercept form and averaged into one left lane and one right lane. This produces a cleaner visual result than drawing every raw segment.

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd adas-system
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add input video

Place a road video at:

```text
data/raw_videos/road.mp4
```

### 5. Run the project

```bash
python src/main.py
```

Press `q` while an OpenCV window is focused to stop playback.

## Current Output Views

The program opens several debug windows:

- `Masked Edges`: Canny edges after applying the road-region mask.
- `Hough Lines`: raw line segments detected by the Hough Transform.
- `Filtered Hough Lines`: lane-like segments after slope and position filtering.
- `Average Lanes`: averaged left and right lane estimates.

These views are useful while learning because they show how each processing step changes the frame.

## Roadmap

### Phase 1: Lane Detection Foundation

- Load and display road footage.
- Apply grayscale conversion and Canny edge detection.
- Build a region-of-interest mask.
- Detect Hough line segments.
- Filter and average lane candidates.
- Overlay final lane lines on the original frame.
- Improve stability across frames.

### Phase 2: Object Detection

- Integrate a pretrained object detector.
- Detect vehicles and pedestrians.
- Draw bounding boxes and labels.
- Estimate object position relative to the lane area.

### Phase 3: Driver Assistance Logic

- Add lane-departure warning logic.
- Add basic collision-risk heuristics.
- Track nearby objects across frames.
- Display warnings in a clean dashboard-style overlay.

### Phase 4: Engineering Polish

- Refactor the prototype into reusable modules.
- Add configuration files for thresholds and model paths.
- Save processed output videos.
- Add tests for utility functions.
- Improve documentation and architecture diagrams.

## Engineering Decisions

- **Classical computer vision first:** Starting with OpenCV fundamentals makes the perception pipeline easier to understand before adding deep learning.
- **Debug windows instead of one final overlay:** Intermediate views make it easier to learn, tune thresholds, and explain the algorithm.
- **Modular project direction:** Lane detection, object detection, and warning logic are separated conceptually so the project can grow without becoming one large script.
- **Pretrained models planned:** The project will use existing object-detection models instead of training custom neural networks, keeping the scope realistic.

## Resume Highlights

This project demonstrates:

- Computer vision preprocessing with OpenCV.
- Real-time frame-by-frame video processing.
- Lane detection using Canny edges and Hough Transforms.
- Geometric filtering based on slope, position, and road-region assumptions.
- Incremental software design for an automotive safety application.
- Clear documentation of technical tradeoffs and future improvements.

## Limitations

- Current lane detection assumes a forward-facing road camera.
- Performance depends on lighting, lane visibility, and road geometry.
- Curved lanes are not fully modeled yet.
- Object detection and collision warnings are planned but not yet implemented.
- This system is for learning and demonstration only, not real vehicle deployment.

## Safety Note

This project is an educational prototype. It should not be used for real-world driving decisions, vehicle control, or safety-critical deployment.

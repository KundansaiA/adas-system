# ADAS Driver Assistance System

## Objective

Develop a computer vision-based Advanced Driver Assistance System (ADAS) capable of analyzing dashcam footage to assist drivers through lane detection, object detection, and collision warnings.

## Features Implemented

### Lane Detection

- Video frame processing using OpenCV
- Canny edge detection
- Region of Interest (ROI) filtering
- Hough Transform lane extraction
- Left and right lane classification
- Lane averaging and temporal smoothing

### Lane Monitoring

- Lane center estimation
- Vehicle center estimation
- Lane offset calculation
- Lane departure status detection

### Object Detection

- YOLOv8 object detection
- Detection of cars, trucks, buses, motorcycles, bicycles, and pedestrians
- Bounding box visualization
- Confidence score display

### Collision Warning System

- Distance classification using bounding box size: Near, Medium, Far
- Path classification using lane position: In Path, Near Path, Outside Path
- Driver warnings: Collision Warning, Pedestrian Warning, Caution

## Technologies Used

- Python
- OpenCV
- NumPy
- Ultralytics YOLOv8
- Git
- GitHub

## Future Improvements

- Object tracking across frames
- Improved distance estimation
- Warning stabilization across multiple frames
- Real-time webcam support
- Lane change detection
- Performance optimization

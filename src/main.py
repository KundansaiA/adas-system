import cv2 # OpenCV library for video processing

from lane_detection import process_lane_detection
from object_detection import process_object_detection

VIDEO_PATH = "data/raw_videos/road.mp4" # Store the video path in one variable so it is easy to check and update


cap = cv2.VideoCapture(VIDEO_PATH) # Open the video file

if not cap.isOpened(): # Stop early if OpenCV could not open the video file
    raise FileNotFoundError(f"Could not open video file: {VIDEO_PATH}")

while True: # Loop to read and display video frames
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or frame could not be read.")
        break

    lane_overlay = process_lane_detection(frame)
    object_overlay = process_object_detection(lane_overlay)

    cv2.imshow("Lane Overlay", lane_overlay) # Display averaged lane endpoints directly on the original road frame
    cv2.imshow("Object Detection", object_overlay) # Display YOLO vehicle and pedestrian detections for visual inspection

    if cv2.waitKey(1) & 0xFF == ord("q"): # Wait for 1 ms and check if the 'q' key is pressed to exit the loop
        break

cap.release() # Release the video capture object to free up resources
cv2.destroyAllWindows() # Close all OpenCV windows to clean up the display after the video is done playing

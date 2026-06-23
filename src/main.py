import cv2 # OpenCV library for video processing

from lane_detection import process_lane_detection
from object_detection import process_object_detection

VIDEO_PATH = "data/raw_videos/road.mp4" # Store the video path in one variable so it is easy to check and update
DISPLAY_WIDTH = 800 # Resize display windows so large video frames fit on screen


cap = cv2.VideoCapture(VIDEO_PATH) # Open the video file

if not cap.isOpened(): # Stop early if OpenCV could not open the video file
    raise FileNotFoundError(f"Could not open video file: {VIDEO_PATH}")


def resize_for_display(image):
    # Resize only the image shown in OpenCV windows; keep processing on the original frame size.
    height, width = image.shape[:2]
    display_height = int(height * (DISPLAY_WIDTH / width))
    return cv2.resize(image, (DISPLAY_WIDTH, display_height))


while True: # Loop to read and display video frames
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or frame could not be read.")
        break

    lane_overlay, lane_info = process_lane_detection(frame, return_info=True)
    cv2.imshow("Lane Overlay", resize_for_display(lane_overlay)) # Display lane output before YOLO starts so a window appears quickly

    if cv2.waitKey(1) & 0xFF == ord("q"): # Give OpenCV time to create/update windows before the slower YOLO step
        break

    object_overlay = process_object_detection(lane_overlay, lane_info)
    cv2.imshow("Object Detection", resize_for_display(object_overlay)) # Display YOLO vehicle and pedestrian detections for visual inspection

    if cv2.waitKey(1) & 0xFF == ord("q"): # Wait for 1 ms and check if the 'q' key is pressed to exit the loop
        break

cap.release() # Release the video capture object to free up resources
cv2.destroyAllWindows() # Close all OpenCV windows to clean up the display after the video is done playing

import cv2 # OpenCV library for image processing and lane overlay drawing
import numpy as np # NumPy is used to store ROI points in the format OpenCV expects

__all__ = ["process_lane_detection"]

_SMOOTHING_PREVIOUS_WEIGHT = 0.8 # Weight for the previous frame's lane estimate during temporal smoothing
_SMOOTHING_CURRENT_WEIGHT = 0.2 # Weight for the current frame's raw lane estimate during temporal smoothing
_LANE_DEPARTURE_THRESHOLD = 100 # Pixel offset from lane center before showing a lane-departure warning
_DISPLAY_WIDTH = 800 # Use a smaller display width so large video frames fit on screen

_previous_left_slope = None # Store the previous smoothed left-lane slope for temporal smoothing
_previous_left_intercept = None # Store the previous smoothed left-lane intercept for temporal smoothing
_previous_right_slope = None # Store the previous smoothed right-lane slope for temporal smoothing
_previous_right_intercept = None # Store the previous smoothed right-lane intercept for temporal smoothing


def _resize_for_display(image):
    # Resize an image for display without changing the full-resolution processing result.
    height, width = image.shape[:2]
    display_height = int(height * (_DISPLAY_WIDTH / width))
    return cv2.resize(image, (_DISPLAY_WIDTH, display_height))


def _make_lane_points(lane_average, height):
    # Convert an averaged lane from slope/intercept form into pixel endpoints that OpenCV can draw later.
    # lane_average contains one representative line for a lane: slope (m) and intercept (b) from y = mx + b.
    slope, intercept = lane_average

    # The bottom endpoint should sit at the bottom of the image, where the lane is closest to the car/camera.
    y_bottom = height

    # The top endpoint stops around 60% down the frame so the final lane line stays inside the road ROI.
    y_top = int(height * 0.6)

    # Rearrange y = mx + b into x = (y - b) / m so we can find the x-coordinate at the bottom y-position.
    x_bottom = int((y_bottom - intercept) / slope)

    # Do the same calculation for the top y-position to get the second endpoint of the lane line.
    x_top = int((y_top - intercept) / slope)

    # Return endpoints in the same order OpenCV line drawing will need: start point then end point.
    return (x_bottom, y_bottom, x_top, y_top)


def _smooth_lane_value(previous_value, current_value):
    # Blend the previous lane estimate with the current estimate to reduce frame-to-frame jitter.
    if previous_value is None:
        return current_value
    return (_SMOOTHING_PREVIOUS_WEIGHT * previous_value) + (_SMOOTHING_CURRENT_WEIGHT * current_value)


def _draw_lane_center_visualization(image, left_lane_average, right_lane_average, width, height):
    # Visualize the lane center at a fixed y-position and return the values other modules need.
    left_slope, left_intercept = left_lane_average
    right_slope, right_intercept = right_lane_average
    center_y = int(height * 0.75)

    # Rearrange y = mx + b into x = (y - b) / m for both final smoothed lane equations.
    left_x_at_center_y = (center_y - left_intercept) / left_slope
    right_x_at_center_y = (center_y - right_intercept) / right_slope
    lane_center_x = (left_x_at_center_y + right_x_at_center_y) / 2
    car_center_x = width / 2
    offset_pixels = car_center_x - lane_center_x

    cv2.line(image, (0, center_y), (width, center_y), (255, 255, 255), 2)
    cv2.line(image, (int(lane_center_x), 0), (int(lane_center_x), height), (0, 255, 0), 3)
    cv2.line(image, (int(car_center_x), 0), (int(car_center_x), height), (255, 0, 0), 3)

    if offset_pixels > _LANE_DEPARTURE_THRESHOLD:
        lane_status = "Drifting Right"
    elif offset_pixels < -_LANE_DEPARTURE_THRESHOLD:
        lane_status = "Drifting Left"
    else:
        lane_status = "Centered"

    cv2.putText(
        image,
        lane_status,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
    )

    print(f"center_y: {center_y}")
    print(f"left_x_at_center_y: {left_x_at_center_y}")
    print(f"right_x_at_center_y: {right_x_at_center_y}")
    print(f"lane_center_x: {lane_center_x}")
    print(f"car_center_x: {car_center_x}")
    print(f"offset_pixels: {offset_pixels}")

    return {
        "center_y": center_y,
        "lane_center_x": lane_center_x,
        "car_center_x": car_center_x,
        "offset_pixels": offset_pixels,
        "lane_status": lane_status,
    }


def process_lane_detection(frame, return_info=False):
    # Run the full lane-detection pipeline for one frame and return the final lane overlay frame.
    global _previous_left_slope
    global _previous_left_intercept
    global _previous_right_slope
    global _previous_right_intercept

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    height, width = frame.shape[:2] # Read the current frame dimensions as (height, width)
    lane_overlay = frame.copy() # Keep an untouched copy of the original frame for the final lane overlay visualization
    roi_points = np.array([ # Define a trapezoid region of interest using percentages of the frame size
        (int(width * 0.02), height),
        (int(width * 0.43), int(height * 0.55)),
        (int(width * 0.57), int(height * 0.55)),
        (int(width * 0.98), height),
    ], dtype=np.int32) #Stores these coordinates as 32-bit integers
    mask = np.zeros_like(edges) # Create a black mask with the same dimensions as the edges image
    cv2.fillPoly(mask, [roi_points], 255) # Fill the defined ROI on the mask with white (255) to create a binary mask
    masked_edges = cv2.bitwise_and(edges, mask) # Apply the mask to the edges image to keep only the edges within the ROI

    hough_lines = np.zeros_like(frame) # Create a blank color image where raw Hough line segments will be drawn
    filtered_hough_lines = np.zeros_like(frame) # Create a second blank image for candidate-filtered Hough line segments
    # Create a separate black image with the same height, width, and color channels as the video frame.
    # This lets us inspect only the averaged lane lines without hiding details behind the original road image.
    average_lanes = np.zeros_like(frame)
    left_lines = [] # Store filtered Hough segments that are candidates for the left lane boundary
    right_lines = [] # Store filtered Hough segments that are candidates for the right lane boundary
    frame_center = width / 2 # Use the image midpoint to reject lane candidates detected on the wrong side
    lines = cv2.HoughLinesP( # Detect straight line segments from the edge pixels that remain inside the ROI
        masked_edges, # The input edge image that has been masked to keep only the edges within the ROI
        rho=1, # Distance resolution of the accumulator in pixels. A value of 1 means that the distance resolution is 1 pixel. (Check every pixel)
        theta=np.pi / 180, #Hough also uses radians not degrees # Angle resolution of the accumulator in radians. A value of np.pi / 180 means that the angle resolution is 1 degree.
        threshold=50, # Minimum number of votes (intersections in Hough space) a line segment needs to be considered valid.
        minLineLength=40, # Minimum length of a line segment to be considered valid.
        maxLineGap=100, # Maximum gap between two points on a line segment to be considered part of the same line.
    )

    if lines is not None: # HoughLinesP returns None when no line segments meet the vote threshold
        for line in lines: # Each line is represented as a 4-element array [x1, y1, x2, y2] where (x1, y1) and (x2, y2) are the endpoints of the line segment
            x1, y1, x2, y2 = line[0] # Each line is represented as a 4-element array [x1, y1, x2, y2]
            if x2 == x1:
                continue # Skip vertical lines to avoid division by zero when calculating slope
            slope = (y2 - y1) / (x2 - x1) # Calculate the slope of the line segment
            if abs(slope) > 0.5: # Filter out line segments with an absolute slope less than or equal to 0.5, which are likely to be horizontal and not lane lines
                print(f"Hough line slope={slope:.2f}, points=({x1}, {y1}) -> ({x2}, {y2})") # Print meaningful-slope lines for debugging without filtering them
                if slope < 0 and x1 < frame_center and x2 < frame_center:
                    left_lines.append((x1, y1, x2, y2)) # Negative-slope segments are left lane candidates
                    cv2.line(filtered_hough_lines, (x1, y1), (x2, y2), (0, 255, 0), 3) # Draw accepted left-lane candidate segments in green
                if slope > 0 and x1 > frame_center and x2 > frame_center:
                    right_lines.append((x1, y1, x2, y2)) # Positive-slope segments are right lane candidates
                    cv2.line(filtered_hough_lines, (x1, y1), (x2, y2), (0, 255, 0), 3) # Draw accepted right-lane candidate segments in green
            cv2.line(hough_lines, (x1, y1), (x2, y2), (0, 0, 255), 3) # (hough line to draw on, start point, end point, color (red), thickness  (3 pixels))

    print(f"Lane candidates: left={len(left_lines)}, right={len(right_lines)}") # Report current-frame candidate counts without averaging or drawing new lane lines

    left_fit = [] # Store (slope, intercept) pairs for left-lane candidate segments
    right_fit = [] # Store (slope, intercept) pairs for right-lane candidate segments
    left_lane_points = None # Store final smoothed/reused left-lane endpoints for center calculation
    right_lane_points = None # Store final smoothed/reused right-lane endpoints for center calculation
    lane_info = None # Store lane-center metadata for downstream ADAS logic when both lanes are available

    for x1, y1, x2, y2 in left_lines:
        slope = (y2 - y1) / (x2 - x1) # Convert the segment endpoints into y = mx + b form
        intercept = y1 - (slope * x1) # Rearranged from y = mx + b, so b = y - mx
        left_fit.append((slope, intercept))

    for x1, y1, x2, y2 in right_lines:
        slope = (y2 - y1) / (x2 - x1) # Convert the segment endpoints into y = mx + b form
        intercept = y1 - (slope * x1) # Rearranged from y = mx + b, so b = y - mx
        right_fit.append((slope, intercept))

    if left_fit:
        left_average_slope, left_average_intercept = np.average(left_fit, axis=0) # Average all left-lane slopes and intercepts into one representative left lane
        left_smoothed_slope = _smooth_lane_value(_previous_left_slope, left_average_slope) # Smooth the left slope against the previous frame's estimate
        left_smoothed_intercept = _smooth_lane_value(_previous_left_intercept, left_average_intercept) # Smooth the left intercept separately from the slope
        _previous_left_slope = left_smoothed_slope # Save the smoothed slope for the next frame
        _previous_left_intercept = left_smoothed_intercept # Save the smoothed intercept for the next frame
        left_lane_points = _make_lane_points((left_smoothed_slope, left_smoothed_intercept), height) # Convert the smoothed left lane into drawable pixel endpoints

        # Draw the averaged left lane on the separate debug image, not on the original frame yet.
        # left_lane_points stores values as (x_bottom, y_bottom, x_top, y_top), so the first point is the bottom endpoint.
        # OpenCV uses BGR color order, so (255, 0, 0) draws blue; thickness 6 makes the averaged line easy to see.
        cv2.line(average_lanes, (left_lane_points[0], left_lane_points[1]), (left_lane_points[2], left_lane_points[3]), (255, 0, 0), 6)
        # Draw the same computed endpoints on top of the original frame copy for the final lane overlay view.
        cv2.line(lane_overlay, (left_lane_points[0], left_lane_points[1]), (left_lane_points[2], left_lane_points[3]), (0, 255, 0), 6)

        print(f"Left raw average: slope={left_average_slope:.2f}, intercept={left_average_intercept:.2f}")
        print(f"Left smoothed average: slope={left_smoothed_slope:.2f}, intercept={left_smoothed_intercept:.2f}")
        print(f"Left lane endpoints: {left_lane_points}")
    else:
        if _previous_left_slope is not None and _previous_left_intercept is not None:
            left_lane_points = _make_lane_points((_previous_left_slope, _previous_left_intercept), height) # Reuse the last smoothed left lane when this frame has no valid left average
            cv2.line(average_lanes, (left_lane_points[0], left_lane_points[1]), (left_lane_points[2], left_lane_points[3]), (255, 0, 0), 6)
            cv2.line(lane_overlay, (left_lane_points[0], left_lane_points[1]), (left_lane_points[2], left_lane_points[3]), (0, 255, 0), 6)

            print("Left average: no lane candidates")
            print(f"Left lane reused from previous frame: slope={_previous_left_slope:.2f}, intercept={_previous_left_intercept:.2f}")
            print(f"Left lane endpoints: {left_lane_points}")
        else:
            print("Left average: no lane candidates and no previous lane to reuse")

    if right_fit:
        right_average_slope, right_average_intercept = np.average(right_fit, axis=0) # Average all right-lane slopes and intercepts into one representative right lane
        right_smoothed_slope = _smooth_lane_value(_previous_right_slope, right_average_slope) # Smooth the right slope against the previous frame's estimate
        right_smoothed_intercept = _smooth_lane_value(_previous_right_intercept, right_average_intercept) # Smooth the right intercept separately from the slope
        _previous_right_slope = right_smoothed_slope # Save the smoothed slope for the next frame
        _previous_right_intercept = right_smoothed_intercept # Save the smoothed intercept for the next frame
        right_lane_points = _make_lane_points((right_smoothed_slope, right_smoothed_intercept), height) # Convert the smoothed right lane into drawable pixel endpoints

        # Draw the averaged right lane on the same debug image as the left averaged lane.
        # The first tuple is the bottom endpoint and the second tuple is the top endpoint.
        # Using the same blue color for both sides shows these are the final averaged lane estimates, not raw Hough segments.
        cv2.line(average_lanes, (right_lane_points[0], right_lane_points[1]), (right_lane_points[2], right_lane_points[3]), (255, 0, 0), 6)
        # Draw the same computed endpoints on the original frame copy so the lane estimate is visible in road context.
        cv2.line(lane_overlay, (right_lane_points[0], right_lane_points[1]), (right_lane_points[2], right_lane_points[3]), (0, 255, 0), 6)

        print(f"Right raw average: slope={right_average_slope:.2f}, intercept={right_average_intercept:.2f}")
        print(f"Right smoothed average: slope={right_smoothed_slope:.2f}, intercept={right_smoothed_intercept:.2f}")
        print(f"Right lane endpoints: {right_lane_points}")
    else:
        if _previous_right_slope is not None and _previous_right_intercept is not None:
            right_lane_points = _make_lane_points((_previous_right_slope, _previous_right_intercept), height) # Reuse the last smoothed right lane when this frame has no valid right average
            cv2.line(average_lanes, (right_lane_points[0], right_lane_points[1]), (right_lane_points[2], right_lane_points[3]), (255, 0, 0), 6)
            cv2.line(lane_overlay, (right_lane_points[0], right_lane_points[1]), (right_lane_points[2], right_lane_points[3]), (0, 255, 0), 6)

            print("Right average: no lane candidates")
            print(f"Right lane reused from previous frame: slope={_previous_right_slope:.2f}, intercept={_previous_right_intercept:.2f}")
            print(f"Right lane endpoints: {right_lane_points}")
        else:
            print("Right average: no lane candidates and no previous lane to reuse")

    if left_lane_points is not None and right_lane_points is not None:
        lane_info = _draw_lane_center_visualization(
            lane_overlay,
            (_previous_left_slope, _previous_left_intercept),
            (_previous_right_slope, _previous_right_intercept),
            width,
            height,
        )

    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 0, 0), thickness=8) # Draw a thick black outline first so the ROI has contrast
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 255), thickness=4) # Draw a bright yellow outline on top for visibility
    #cv2.imshow("Frame with ROI", frame) # Display the original frame with the ROI outline drawn on top
    cv2.imshow("Masked Edges", _resize_for_display(masked_edges)) # Display the edges that are within the ROI in a window named "Masked Edges"
    cv2.imshow("Hough Lines", _resize_for_display(hough_lines)) # Display the raw line segments detected by the Hough transform
    cv2.imshow("Filtered Hough Lines", _resize_for_display(filtered_hough_lines)) # Display Hough segments that pass both slope and x-position filters
    # Display the averaged-lane debug image in its own window.
    # This keeps the averaged result separate from the raw frame so we can verify the line positions before overlaying them later.
    cv2.imshow("Average Lanes", _resize_for_display(average_lanes))

    if return_info:
        return lane_overlay, lane_info

    return lane_overlay

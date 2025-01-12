# Import necessary libraries
import numpy as np
import cv2 
from ultralytics import YOLO  
import cvzone  # Helper library for drawing annotations on frames
import logging
from datetime import datetime

# Configration the logging system 
logging.basicConfig(
    filename="detection_events.log",
    filemode="w",  # Open the file in write mode (clears the file on each run)
    level = logging.INFO,
    format = "%(asctime)s - %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S"
)

# Load the pre-trained YOLO model
model = YOLO("best.pt")  # Replace "best.pt" with the path to your trained model
class_names = model.names  # Get class names from the model (e.g., "person", "car", etc.)

# Open the video file for processing
video_capture = cv2.VideoCapture("sample.mp4")  # Replace "sample.mp4" with your video file path

# Main loop to process each frame of the video
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    if not ret:  # If no frame is read (end of video), break the loop
        break

    # Resize the frame to a fixed size for consistent processing
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection and tracking using the YOLO model
    detection_results = model.track(frame, persist=True)  # `persist=True` ensures tracking IDs are consistent across frames

    # Check if any objects are detected in the frame
    if detection_results[0].boxes is not None:
        # Extract bounding box coordinates and convert them to a list of integers
        bounding_boxes = detection_results[0].boxes.xyxy.int().cpu().tolist()
        # Extract class IDs for detected objects
        class_ids = detection_results[0].boxes.cls.int().cpu().tolist()

        # Extract tracking IDs if available (for tracked objects)
        if detection_results[0].boxes.id is not None:
            tracking_ids = detection_results[0].boxes.id.int().cpu().tolist()
        else:
            # If no tracking IDs are available, assign -1 as a placeholder
            tracking_ids = [-1] * len(bounding_boxes)

        # Extract segmentation masks if available
        segmentation_masks = detection_results[0].masks
        if segmentation_masks is not None:
            # Convert masks to a list of polygon points
            segmentation_masks = segmentation_masks.xy
            # Create a copy of the frame for overlay purposes
            overlay_frame = frame.copy()

            # Loop through each detected object (bounding box, tracking ID, class ID, and mask)
            for bounding_box, tracking_id, class_id, mask_points in zip(bounding_boxes, tracking_ids, class_ids, segmentation_masks):
                x_min, y_min, x_max, y_max = bounding_box  # Extract bounding box coordinates
                class_name = class_names[class_id]  # Get the class name (e.g., "person", "car")

                # Log the detection event to a log file
                logging.info(f"Object ID: {tracking_id}, Class: {class_name}, Bounding Box: ({x_min}, {y_min},{x_max}, {y_max})")

                # Check if the mask has valid points
                if mask_points.size > 0:
                    # Reshape the mask points for drawing
                    mask_points = np.array(mask_points, dtype=np.int32).reshape((-1, 2, 1))

                    # Draw the bounding box on the frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    # Fill the mask area on the overlay with a color (e.g., red)
                    cv2.fillPoly(overlay_frame, [mask_points], color=(0, 0, 255))

                    # Add text annotations for tracking ID and class name
                    cvzone.putTextRect(frame, f"ID: {tracking_id}", (x_max, y_max), scale=1, thickness=1)
                    cvzone.putTextRect(frame, f"{class_name}", (x_min, y_min), scale=1, thickness=1)

            # Blend the overlay with the original frame for a transparent effect
            transparency_alpha = 0.5  # Transparency factor (0 = fully transparent, 1 = fully opaque)
            frame = cv2.addWeighted(overlay_frame, transparency_alpha, frame, 1 - transparency_alpha, 0)

    # Display the processed frame in a window
    cv2.imshow("Detection & Segmentation", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
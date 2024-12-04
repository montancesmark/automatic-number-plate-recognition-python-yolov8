import time
import numpy as np
from ultralytics import YOLO
import cv2

import util 
#from sort.sort import *
from util import get_car, read_license_plate, write_csv

# Configurable Parameters
frame_time_threshold = 1.0  # Threshold for warning if frame processing time exceeds this value (seconds)
processing_secs = 10  # Number of seconds to process

# Initialize models with error handling
try:
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Load video with error handling
video_path = './sample.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Dynamically retrieve FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    print("Error: Could not retrieve FPS from video.")
    exit()
print(f"Video FPS: {fps}")

# Compute the total number of frames to process
frames_to_process = int(fps * processing_secs)

results = {}
#mot_tracker = Sort()
vehicles = [2, 3, 5, 7]  # Vehicle class IDs (e.g., car, truck, etc.)

# Initialize timer for total processing time
start_time = time.time()
frame_times = []  # List to track per-frame processing times

frame_nmr = -1
ret = True

# Read and process frames
while ret and frame_nmr < frames_to_process:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # Start timer for each frame
        frame_start_time = time.time()

        results[frame_nmr] = {}
        try:
            # Detect vehicles
            detections = coco_model.track(frame, verbose=False, persist=True)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, track_id, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, track_id])

            # Detect license plates
            license_plates = license_plate_detector(frame, verbose=False)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, detections_)

                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                    # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(
                        license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                    )

                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(
                        license_plate_crop_thresh
                    )

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score,
                            },
                        }
        except Exception as e:
            print(f"Error processing frame {frame_nmr}: {e}")

        # End frame timer and record the frame time
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        frame_times.append(frame_time)

        # Log any frame that exceeds the processing time threshold
        if frame_time > frame_time_threshold:
            print(f"Warning: Frame {frame_nmr} took {frame_time:.2f} seconds to process!")

# Calculate and print the total processing time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total processing time for all frames: {elapsed_time:.2f} seconds")

# Calculate and print average frame processing time
avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
print(f"Average frame processing time: {avg_frame_time:.2f} seconds")

# Write results
try:
    write_csv(results, './test.csv')
    print("Results written to CSV successfully.")
except Exception as e:
    print(f"Error writing results to CSV: {e}")

# Release video capture object
cap.release()

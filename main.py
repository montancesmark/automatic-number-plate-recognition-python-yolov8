import time 

from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

#Note: change these as needed
fps = 30    # actual fps
secs = 5    # number of seconds to process


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]

# Initialize timer for total processing time
start_time = time.time()
frame_times = []  # List to track per-frame processing times

# read frames
frame_nmr = -1
ret = True
while ret and frame_nmr < fps * secs:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # Start timer for each frame
        frame_start_time = time.time()

        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame, verbose=False)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame, verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

        # End frame timer and record the frame time
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        frame_times.append(frame_time)

        # You can set a threshold, e.g., 1 second per frame, to log any frame that takes too long
        if frame_time > 1.0:  # 1 second is just an example threshold
            print(f"Warning: Frame {frame_nmr} took {frame_time:.2f} seconds to process!")

# Calculate and print the total processing time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total processing time for all frames: {elapsed_time:.2f} seconds")

# Calculate and print average frame processing time
avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
print(f"Average frame processing time: {avg_frame_time:.2f} seconds")

# write results
write_csv(results, './test.csv')

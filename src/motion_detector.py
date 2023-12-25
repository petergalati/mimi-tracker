import cv2 as cv
import numpy as np
import utils





# import tensorflow.

from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# setup stuff

base_options=core.BaseOptions(
    file_name="/home/peterg/mimi-tracker/models/1.tflite"
)

detection_options = processor.DetectionOptions(
    max_results=2,
    score_threshold=0.3
)

object_options = vision.ObjectDetectorOptions(
    base_options=base_options,
    detection_options=detection_options
)

detector = vision.ObjectDetector.create_from_options(object_options)




cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv.flip(frame, 1)

    # Our operations on the frame come here
    rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    detection_result = detector.detect(input_tensor)
    
    frame = utils.visualise(frame, detection_result)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
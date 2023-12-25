import cv2 as cv
import numpy as np
from tflite_support.task import processor

def visualise(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
):
    for detection in detection_result:
        box = detection.bounding_box
        start = box.origin_x, box.origin_y
        end = box.origin_x + box.width, box.origin_y + box.height
        cv.rectangle(image, start, end, (255, 0, 0), 2)

        category = detection.categories[0]
        category_name = category.category_name
        p = round(category.score, 2)
        label = f"{category_name} + ' (' + {str(p)} + ')'"
        label_location = (box.origin_x, box.origin_y + 20)
        cv.putText(image, label, label_location, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


    
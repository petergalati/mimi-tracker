import cv2 as cv
import torch

def determine_direction(prev_x_center, x_center, sensitivity):
    """
    Determine the movement direction based on the current and previous x_center positions.
    """
    if prev_x_center is not None and x_center is not None:
        if abs(x_center - prev_x_center) > sensitivity:
            if x_center > prev_x_center:
                return "Right"
            elif x_center < prev_x_center:
                return "Left"
        else:
            return "Stationary"
    return "No Movement"


model = torch.hub.load('yolov5', 'custom', path='yolov5n.pt', source='local')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Variables to track previous position
prev_x_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    # Initialize variables to track the position of the detected person
    x_center = None

    # Draw bounding boxes for detected objects
    for i in range(len(labels)):
        if int(labels[i]) == 0:  # YOLOv5 COCO class ID for cat is 15
            x1, y1, x2, y2, conf = cords[i]
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
            
            # Calculate the center of the bounding box
            x_center = (x1 + x2) / 2
            
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f'Mimi {conf:.2f}', (x1, y1 - 10), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    
    # Determine movement direction
    sensitivity = 50  # Adjust this value to make movement detection more or less sensitive
    movement_direction = determine_direction(prev_x_center, x_center, sensitivity)
    print(f"Movement Direction: {movement_direction}")

    # Update previous center position
    prev_x_center = x_center

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


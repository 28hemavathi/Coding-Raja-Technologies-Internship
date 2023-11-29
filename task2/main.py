import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Load video
cap = cv2.VideoCapture("input_video.mp4")

# Initialize tracker
tracker = cv2.TrackerCSRT_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Tracking
    boxes = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0: # Assuming class 0 is for person
                center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append((x, y, w, h))
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Initialize tracker with the bounding box
        tracker.init(frame, (x, y, w, h))

    # Update tracker
    success, boxes = tracker.update(frame)

    for new_box in boxes:
        (x, y, w, h) = [int(v) for v in new_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


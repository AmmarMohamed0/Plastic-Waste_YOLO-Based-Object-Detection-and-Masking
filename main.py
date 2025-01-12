import numpy as np
import cv2
from ultralytics import YOLO
import cvzone 

model = YOLO("best.pt")
names = model.names

cap = cv2.VideoCapture("sample.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(1020,500))

    results = model.track(frame, persist=True)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist() 
        class_ids = results[0].boxes.cls.int().cpu().tolist() 

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1]*len(boxes)
        
        masks = results[0].masks
        if masks is not None:
            clss = results[0].boxes.cls.int().cpu().tolist()
            masks = masks.xy
            overlay = frame.copy()

            for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                x1, y1, x2, y2 = box
                c = names[class_id]

                if mask.size > 0 :
                    mask = np.array(mask, dtype=np.int32).reshape((-1, 2, 1))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.fillPoly(overlay, [mask], color = (0,0,255))

                    cvzone.putTextRect(frame, f"{track_id}",(x2, x2), 1, 1)
                    cvzone.putTextRect(frame, f"{c}", (x1, y1), 1, 1)

            alpha = 0.5  # Transparency factor (0 = invisible, 1 = fully visible)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.imshow("Detec&Mask", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
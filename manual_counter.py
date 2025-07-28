from ultralytics import YOLO
from sort import Sort
import numpy as np
import cv2
import os

def main():
    path_to_model = os.path.join(os.getcwd(), 'Potato_model_best_weights.pt')
    path_to_video = os.path.join(os.getcwd(),'Inference_data', 'potato.mp4')


    source = cv2.VideoCapture(path_to_video)
    cv2.namedWindow("Potato Detector", cv2.WINDOW_NORMAL)

    model = YOLO(path_to_model)
    line = [850,100,850,650]
    tracker = Sort(max_age=10)
    CountedIDs = set()


    while source.isOpened():
        detections = np.empty((0, 5))
        success, frame = source.read()

        if not success:
            print("Video frame is empty or processing is complete.")
            break
        
        result = model(frame, stream=True)
        result = next(result)

        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]
            label = model.names[int(box.cls[0])]
            detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            

        output = tracker.update(detections)
        for detection in output:
            x1, y1, x2, y2, track_id = detection
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if line[1] < cy < line[3] and line[0]-20 < cx < line[0]+20:
                CountedIDs.add(track_id)
                

        cv2.putText(frame, f"Counted IDs: {len(CountedIDs)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
        print(f"Counted IDs: {CountedIDs}")
        cv2.imshow("Potato Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




import cv2
import torch

def detect_objects(input_path, output_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        for det in detections:
            x1, y1, x2, y2, _, cls = det
            label = results.names[int(cls)]
            if label in ["person", "dog", "cow", "horse", "pig"]:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

import cv2
import torch

def person_only(input_path, output_path):
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

        mask = frame.copy() * 0  # 검정 배경 생성
        for det in detections:
            x1, y1, x2, y2, _, cls = det
            label = results.names[int(cls)]
            if label == "person":
                mask[int(y1):int(y2), int(x1):int(x2)] = frame[int(y1):int(y2), int(x1):int(x2)]

        out.write(mask)

    cap.release()
    out.release()

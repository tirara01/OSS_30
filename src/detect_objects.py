import cv2
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

def detect_objects(input_path, output_path):
    # TensorFlow Hub에서 YOLO 모델 로드
    model_url = "https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1"
    detector = hub.load(model_url)

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 입력 데이터를 모델에 맞게 전처리
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = tf.image.resize(input_tensor, (512, 512))
        input_tensor = tf.expand_dims(input_tensor, axis=0) / 255.0

        # 모델을 사용해 객체 탐지
        detections = detector(input_tensor)
        boxes = detections["detection_boxes"][0].numpy()
        classes = detections["detection_classes"][0].numpy().astype(int)
        scores = detections["detection_scores"][0].numpy()

        # 탐지된 객체를 그리기
        for i, box in enumerate(boxes):
            if scores[i] < 0.5:  # 50% 이상의 확률만 표시
                continue

            y1, x1, y2, x2 = box
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            label = f"Class {classes[i]}: {int(scores[i] * 100)}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

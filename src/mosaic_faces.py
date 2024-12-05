import cv2

def mosaic_faces(input_path, output_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # 모자이크 처리
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y + h, x:x + w] = face

        out.write(frame)

    cap.release()
    out.release()


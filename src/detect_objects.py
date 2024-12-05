import cv2

def detect_objects(input_path, output_path, format='mp4'):
    person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if format == 'mp4':
        output_path += ".mp4"
        codec = cv2.VideoWriter_fourcc(*'mp4v')
    elif format == 'avi':
        output_path += ".avi"
        codec = cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError("지원되지 않는 형식입니다. 'mp4' 또는 'avi'를 사용하세요.")

    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in persons:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

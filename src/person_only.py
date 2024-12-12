import cv2

def person_only(input_path, output_path, format='mp4'):
    # Load Haar cascade for full-body detection
    person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open input video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 형식에 따른 코덱 선택
    # Select codec based on the format
    if format == 'mp4':
        output_path += ".mp4"
        output_file = f"{output_path}.mp4"
        codec = cv2.VideoWriter_fourcc(*'mp4v')
    elif format == 'avi':
        output_path += ".avi"
        output_file = f"{output_path}.avi"
        codec = cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError("지원되지 않는 형식입니다. 'mp4' 또는 'avi'를 사용하세요.")
        raise ValueError("Unsupported format. Use 'mp4' or 'avi'.")

    out = cv2.VideoWriter(output_path, codec, fps, (width, height))
    # Initialize video writer
    out = cv2.VideoWriter(output_file, codec, fps, (width, height))

    while cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 검정 배경 생성
        # Create a black mask
        mask = frame.copy() * 0
        for (x, y, w, h) in persons:
            mask[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
@@ -37,3 +43,4 @@ def person_only(input_path, output_path, format='mp4'):

    cap.release()
    out.release()
    print(f"Processing completed. Saved to {output_file}")

import cv2

def mosaic_faces(input_path, output_path, format='mp4'):
    # 얼굴 검출용 Haar Cascade 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # 비디오 파일 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"입력 파일을 열 수 없습니다: {input_path}")

    # 비디오 정보 얻기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if format.lower() == 'mp4':
        output_path = output_path if output_path.endswith('.mp4') else output_path + ".mp4"
        codec = cv2.VideoWriter_fourcc(*'mp4v')
    elif format.lower() == 'avi':
        output_path = output_path if output_path.endswith('.avi') else output_path + ".avi"
        codec = cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError("지원되지 않는 형식입니다. 'mp4' 또는 'avi'를 사용하세요.")

    # 출력 비디오 파일 생성
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = face

        out.write(frame)

    cap.release()
    out.release()
    print(f"처리 완료! 출력 파일: {output_path}")

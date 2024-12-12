import cv2

def mosaic_faces(input_path, output_path, format='mp4'):
    # 얼굴 검출용 Haar Cascade 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # 비디오 파일 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"입력 파일을 열 수 없습니다: {input_path}")

    # 비디오 정보 얻기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if format == 'mp4':
        output_path += ".mp4"
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        codec = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 포맷
    elif format == 'avi':
        output_path += ".avi"
        codec = cv2.VideoWriter_fourcc(*'XVID')
        codec = cv2.VideoWriter_fourcc(*'XVID')  # AVI 포맷
    else:
        raise ValueError("지원되지 않는 형식입니다. 'mp4' 또는 'avi'를 사용하세요.")


    # 출력 비디오 파일 생성
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

            break  # 비디오 끝에 도달하면 종료

        # 얼굴 검출을 위해 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            # 얼굴 부분을 추출
            face = frame[y:y+h, x:x+w]

            # 얼굴에 모자이크 적용 (10x10 크기 축소 후 다시 원래 크기로 확대)
            face = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y + h, x:x + w] = face

            # 원본 이미지에 모자이크 처리된 얼굴 덮어쓰기
            frame[y:y+h, x:x+w] = face

        # 모자이크 처리된 프레임을 출력 비디오에 저장
        out.write(frame)

    # 리소스 해제
    cap.release()
    out.release()
    print(f"처리 완료! 출력 파일: {output_path}")

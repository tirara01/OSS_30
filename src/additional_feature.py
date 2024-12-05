import cv2

def detect_motion(input_path, output_path, format='mp4'):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 형식에 따른 코덱 선택
    if format == 'mp4':
        output_path += ".mp4"
        codec = cv2.VideoWriter_fourcc(*'mp4v')
    elif format == 'avi':
        output_path += ".avi"
        codec = cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError("지원되지 않는 형식입니다. 'mp4' 또는 'avi'를 사용하세요.")

    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # 움직임이 있는 부분을 색상으로 표시
        out.write(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
        prev_gray = gray

    cap.release()
    out.release()

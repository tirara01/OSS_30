import cv2

def detect_motion(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # 기본값 설정
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 색상으로 출력
        out.write(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
        prev_gray = gray

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Motion detection completed and saved to:", output_path)

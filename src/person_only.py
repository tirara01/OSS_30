import cv2
def person_only(input_path, output_path, format='mp4'):
    # Load Haar cascade for full-body detection
    person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade>

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

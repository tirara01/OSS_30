import argparse
from src.detect_objects import detect_objects

def main():
    parser = argparse.ArgumentParser(description="VideoAnalyzer Project")
    parser.add_argument("task", choices=["detect"], help="Task to perform")
    args = parser.parse_args()

    input_video = "assets/sample_video.mp4"

    if args.task == "detect":
        print("Running object detection...")
        detect_objects(input_video, "output_detect.mp4")
    else:
        print("Invalid task.")

if __name__ == "__main__":
    main()

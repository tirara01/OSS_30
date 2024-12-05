import argparse
from src.detect_objects import detect_objects
from src.mosaic_faces import mosaic_faces
from src.person_only import person_only
from src.additional_feature import detect_motion

def main():
    parser = argparse.ArgumentParser(description="VideoAnalyzer Project using OpenCV")
    parser.add_argument("task", choices=["detect", "mosaic", "person", "motion"], help="Task to perform")
    parser.add_argument("--format", choices=["mp4", "avi"], default="mp4", help="Output file format")
    args = parser.parse_args()

    input_video = "assets/sample_video.mp4"

    if args.task == "detect":
        print(f"Running object detection and saving as {args.format.upper()}...")
        detect_objects(input_video, "output_detect", format=args.format)
    elif args.task == "mosaic":
        print(f"Applying face mosaic and saving as {args.format.upper()}...")
        mosaic_faces(input_video, "output_mosaic", format=args.format)
    elif args.task == "person":
        print(f"Filtering only person objects and saving as {args.format.upper()}...")
        person_only(input_video, "output_person", format=args.format)
    elif args.task == "motion":
        print(f"Detecting motion and saving as {args.format.upper()}...")
        detect_motion(input_video, "output_motion", format=args.format)
    else:
        print("Invalid task selected.")

if __name__ == "__main__":
    main()


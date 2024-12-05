import argparse
from src.detect_objects import detect_objects
from src.mosaic_faces import mosaic_faces
from src.person_only import person_only
from src.additional_feature import detect_motion

def main():
    parser = argparse.ArgumentParser(description="VideoAnalyzer Project")
    parser.add_argument("task", choices=["detect", "mosaic", "person", "motion"], help="Task to perform")
    args = parser.parse_args()

    input_video = "assets/sample_video.mp4"

    if args.task == "detect":
        print("Running object detection...")
        detect_objects(input_video, "output_detect.mp4")
    elif args.task == "mosaic":
        print("Applying face mosaic...")
        mosaic_faces(input_video, "output_mosaic.mp4")
    elif args.task == "person":
        print("Filtering only person objects...")
        person_only(input_video, "output_person.mp4")
    elif args.task == "motion":
        print("Detecting motion in video...")
        detect_motion(input_video, "output_motion.mp4")
    else:
        print("Invalid task. Use one of: detect, mosaic, person, motion")

if __name__ == "__main__":
    main()

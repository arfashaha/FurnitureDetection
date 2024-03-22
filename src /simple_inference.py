import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import torch


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    print(torch.cuda.is_available())
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    start_time = 5 # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.engine", task="detect") #change to .pt if using pytorch, since engine is tensorRT format

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

  
    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        labels = [
            result.names[class_id]
            for class_id
            in detections.class_id
        ]		

        frame = box_annotator.annotate(
                    scene=frame, 
                    detections=detections, 
                    labels=labels
                )
        # Show the Frame
        cv2.imshow("yolov8", frame)
        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()

# from ultralytics import YOLO

# # Load a model
# model = YOLO('C:/Users/ASUS/Documents/Kuliah/Semester 7/Skripsi/Codingan Pengujian Model/Test FPS video/src/models/best_X.pt')  # load an official model
# # model = YOLO('path/to/best.pt')  # load a custom trained model

# # Export the model
# model.export(format='engine', device=0, half=True, imgsz=640)

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("C:/Users/ASUS/Documents/Kuliah/Semester 7/Skripsi/Hasil_YOLOv8M_FINAL DATA BENER/content/runs/detect/train/weights/best.pt")

# Define path to video file
source = "C:/Users/ASUS/Documents/Kuliah/Semester 7/Skripsi/Video Testing/Toko Informa.mp4"

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects
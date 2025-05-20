from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')

results = model("https://ultralytics.com/images/bus.jpg")


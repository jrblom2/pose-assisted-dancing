import cv2
from ultralytics import YOLO


model = YOLO('yolo11n-pose.pt')
vidObj = cv2.VideoCapture(6)
success = True
while success:    
    success, image = vidObj.read()   
    result = model(image)
    annotated_image = result[0].plot()

    cv2.imshow("", annotated_image)     
    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

cv2.destroyAllWindows()


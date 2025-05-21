import cv2
from yoloCompare import yoCompare
import numpy as np


yolo = yoCompare()
vidObj = cv2.VideoCapture(6)
success = True
while success:    
    success, image = vidObj.read()   
    result = yolo.detect(image)
    annotated_image = yolo.annotate_image(result)
    keypoints = result[0].keypoints.xy.cpu().numpy()

    if len(keypoints) == 2:
        print(yolo.compare_detections(keypoints[0], keypoints[1]))

    cv2.imshow("", annotated_image)     
    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

cv2.destroyAllWindows()


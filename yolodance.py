import cv2
from yoloCompare import yoCompare


yolo = yoCompare()
vidObj = cv2.VideoCapture(6)
success = True
while success:    
    success, image = vidObj.read()   
    result = yolo.detect(image)
    annotated_image = yolo.annotate_image(result)

    cv2.imshow("", annotated_image)     
    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

cv2.destroyAllWindows()


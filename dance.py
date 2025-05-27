import cv2
from mediaPipeCompare import mpCompare

vidObj = cv2.VideoCapture(6)
mpCompare = mpCompare()
success = True
while success:
    success, image = vidObj.read()
    sets = mpCompare.detect(image)
    annotated_image = mpCompare.draw_landmarks_on_image(image, sets)
    if len(sets) == 2:
        print(mpCompare.compare_detections(sets[0], sets[1]))

    cv2.imshow('test', annotated_image)
    cv2.waitKey(1)

cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity


class yoCompare:

    def __init__(self):
        self.model = YOLO('yolo11n-pose.pt')

    def detect(self, image):
        result = self.model(image, verbose=False)
        return result

    def annotate_image(self, results):
        annotated_image = results[0].plot()
        return annotated_image
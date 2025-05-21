import cv2
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class yoCompare:

    def __init__(self):
        self.model = YOLO('yolo11n-pose.pt')

    def detect(self, image):
        result = self.model(image, verbose=False)
        return result

    def annotate_image(self, results):
        annotated_image = results[0].plot()
        return annotated_image
    
    def compare_detections(self, first, second):
        # flatten
        firstFlat = np.array([coord for lm in first for coord in (lm[0], lm[1])])
        secondFlat = np.array([coord for lm in second for coord in (lm[0], lm[1])])

        # Pay attention to data type here, it is meant to do many comparisons at once so output is a table
        return cosine_similarity(firstFlat.reshape(1, -1), secondFlat.reshape(1, -1))[0, 0]
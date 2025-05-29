from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def normalize(landmarks):
    # Extract all x, y, z into arrays
    xs = np.array([lm[0] for lm in landmarks])
    ys = np.array([lm[1] for lm in landmarks])

    # Bounding box around x, y
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    width = max_x - min_x
    height = max_y - min_y
    scale = max(width, height)  # Preserve aspect ratio

    # Prevent division by zero
    if scale == 0:
        scale = 1e-6

    # Normalize coordinates relative to bounding box
    norm_xs = (xs - min_x) / scale
    norm_ys = (ys - min_y) / scale

    # Return flattened array: [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]
    return np.array([coord for coord in zip(norm_xs, norm_ys)]).flatten()


class yoCompare:

    def __init__(self):
        self.model = YOLO('yolo11n-pose.pt')

    def detect(self, image):
        result = self.model(image, verbose=False)
        return result

    def draw_landmarks_on_image(self, frame, results):
        annotated_image = results[0].plot()
        return annotated_image

    def compare_detections(self, first, second):
        scores = []
        first = first[0].keypoints.xy[0].cpu().numpy()
        for detection in second[0].keypoints.xy:
            second = detection.cpu().numpy()
            filteredFirst = []
            filteredSecond = []
            for f, s in zip(first, second):
                if f[0] != 0 and f[1] != 0 and s[0] != 0 and s[1] != 0:
                    filteredFirst.append(f)
                    filteredSecond.append(s)
            # flatten
            firstFlat = normalize(filteredFirst)
            secondFlat = normalize(filteredSecond)
            # Pay attention to data type here, it is meant to do many comparisons at once so output is a table
            score = round(cosine_similarity(firstFlat.reshape(1, -1), secondFlat.reshape(1, -1))[0, 0], 2)
            scores.append(score)
            filteredFirst.clear()
            filteredSecond.clear()
        return scores

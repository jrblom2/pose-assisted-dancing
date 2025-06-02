from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import cv2
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


COCO_CONNECTIONS = [
    (5, 7),
    (7, 9),  # Left arm
    (6, 8),
    (8, 10),  # Right arm
    (5, 6),  # Shoulders
    (11, 13),
    (13, 15),  # Left leg
    (12, 14),
    (14, 16),  # Right leg
    (11, 12),  # Hips
    (5, 11),
    (6, 12),  # Torso sides
]


class yoCompare:

    def __init__(self):
        self.model = YOLO('yolo11n-pose.pt')

    def detect(self, image):
        result = self.model(image, verbose=False)
        return result

    def draw_landmarks_on_image(self, frame, results):
        annotated_image = frame.copy()

        for result in results:
            if result.keypoints is not None:
                for keypoints in result.keypoints.data:
                    if not len(keypoints) > 0:
                        continue
                    # Draw keypoints
                    for x, y, conf in keypoints:
                        if conf > 0.5:
                            cv2.circle(annotated_image, (int(x), int(y)), 3, (0, 255, 0), -1)

                    # Draw connections
                    for start_idx, end_idx in COCO_CONNECTIONS:
                        x1, y1, c1 = keypoints[start_idx]
                        x2, y2, c2 = keypoints[end_idx]
                        if c1 > 0.5 and c2 > 0.5:
                            cv2.line(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

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

            if len(filteredFirst) < 1 or len(filteredSecond) < 1:
                continue

            # flatten
            firstFlat = normalize(filteredFirst)
            secondFlat = normalize(filteredSecond)
            # Pay attention to data type here, it is meant to do many comparisons at once so output is a table
            score = round(cosine_similarity(firstFlat.reshape(1, -1), secondFlat.reshape(1, -1))[0, 0], 2)
            scores.append(score)
        return scores

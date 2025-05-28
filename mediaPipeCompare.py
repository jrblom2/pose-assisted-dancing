import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2


def normalize(landmarks):
    # Extract all x, y, z into arrays
    xs = np.array([lm.x for lm in landmarks])
    ys = np.array([lm.y for lm in landmarks])

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


class mpCompare:

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True, num_poses=2)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(detection_result)):
            pose_landmarks = detection_result[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in pose_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated_image

    def compare_detections(self, first, second):
        filteredFirst = []
        filteredSecond = []
        for f, s in zip(first, second):
            if f.x != 0 and f.y != 0 and s.x != 0 and s.y != 0:
                filteredFirst.append(f)
                filteredSecond.append(s)
        # flatten
        firstFlat = normalize(first)
        secondFlat = normalize(second)

        # Pay attention to data type here, it is meant to do many comparisons at once so output is a table
        return cosine_similarity(firstFlat.reshape(1, -1), secondFlat.reshape(1, -1))[0, 0]

    def detect(self, image):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detection_result = self.detector.detect(image)

        return detection_result.pose_landmarks

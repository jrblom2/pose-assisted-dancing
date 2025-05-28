import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_thickness = 3
text_color = (0, 255, 0)  # Green (BGR format)
text_position = (50, 50)  # (x, y) coordinates (top-left corner)


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

    def pose_compare(self):
        vidObj = cv2.VideoCapture(6)
        success = True
        while success:
            success, image = vidObj.read()
            sets = self.detect(image)
            annotated_image = self.draw_landmarks_on_image(image, sets)
            if len(sets) == 2:
                score = self.compare_detections(sets[0], sets[1])
                text = f'Score: {score:.2f}'
                cv2.putText(
                    annotated_image, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA
                )
            cv2.imshow('test', annotated_image)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    def dance_compare(self, video_path):
        # Load video and start stream
        vid = cv2.VideoCapture(video_path)
        # img = load_image('dance_videos/test.jpg')
        stream = cv2.VideoCapture(6)

        # Execute models
        vidSuccess = True
        streamSuccess = True
        while streamSuccess:
            vidSuccess, vid_image = vid.read()
            streamSuccess, stream_image = stream.read()

            # Resize images to the same size
            if not vidSuccess or not streamSuccess:
                break
            vid_frame, stream_frame = self.resize_images(vid_image, stream_image)

            vid_sets = self.detect(vid_frame)
            stream_sets = self.detect(stream_frame)
            vid_annotated_image = self.draw_landmarks_on_image(vid_frame, vid_sets)
            stream_annotated_image = self.draw_landmarks_on_image(stream_frame, stream_sets)
            # Combine frames and add score text
            score = self.compare_detections(vid_sets[0], stream_sets[0])
            text = f'Score: {score}'
            frame = np.hstack((vid_annotated_image, stream_annotated_image))
            cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            cv2.imshow('', frame)

            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
                break

    def resize_images(self, vid_image, stream_image):
        height = min(vid_image.shape[0], stream_image.shape[0])
        vid_frame = cv2.resize(vid_image, (int(vid_image.shape[1] * height / vid_image.shape[0]), height))
        stream_frame = cv2.resize(stream_image, (int(stream_image.shape[1] * height / stream_image.shape[0]), height))
        return vid_frame, stream_frame

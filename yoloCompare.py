import cv2
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Variables for text
score = 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_thickness = 3
text_color = (0, 255, 0)  # Green (BGR format)
text_position = (50, 50)  # (x, y) coordinates (top-left corner)


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

    def annotate_image(self, results):
        annotated_image = results[0].plot()
        return annotated_image

    def compare_detections(self, first, second):
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
        return cosine_similarity(firstFlat.reshape(1, -1), secondFlat.reshape(1, -1))[0, 0]

    def pose_compare(self):
        vidObj = cv2.VideoCapture(6)
        success = True
        while success:
            success, image = vidObj.read()
            result = self.detect(image)
            annotated_image = self.annotate_image(result)
            keypoints = result[0].keypoints.xy.cpu().numpy()

            if len(keypoints) == 2:
                print(self.compare_detections(keypoints[0], keypoints[1]))

            cv2.imshow("", annotated_image)

            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
                break

        cv2.destroyAllWindows()

    def dance_compare(self, video_path):
        # Load video and start stream
        vid = cv2.VideoCapture(video_path)
        # img = load_image('dance_videos/test.jpg')
        stream = cv2.VideoCapture(6)

        # Execute models
        vidSuccess = True
        streamSuccess = True
        while vidSuccess and streamSuccess:
            vidSuccess, vid_image = vid.read()
            streamSuccess, stream_image = stream.read()

            # Resize images to the same size
            if not vidSuccess or not streamSuccess:
                break
            vid_frame, stream_frame = self.resize_images(vid_image, stream_image)

            vid_result = self.detect(vid_frame)
            stream_result = self.detect(stream_frame)
            vid_annotated_image = self.annotate_image(vid_result)
            stream_annotated_image = self.annotate_image(stream_result)
            vid_keypoints = vid_result[0].keypoints.xy[0].cpu().numpy()
            stream_keypoints = stream_result[0].keypoints.xy[0].cpu().numpy()

            # Combine frames and add score text
            score = self.compare_detections(vid_keypoints, stream_keypoints)
            text = f'Score: {score}'
            frame = np.hstack((vid_annotated_image, stream_annotated_image))
            cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            cv2.imshow("", frame)

            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
                break

    def resize_images(self, vid_image, stream_image):
        height = min(vid_image.shape[0], stream_image.shape[0])
        vid_frame = cv2.resize(vid_image, (int(vid_image.shape[1] * height / vid_image.shape[0]), height))
        stream_frame = cv2.resize(stream_image, (int(stream_image.shape[1] * height / stream_image.shape[0]), height))
        return vid_frame, stream_frame

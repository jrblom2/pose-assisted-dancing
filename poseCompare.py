import cv2
import numpy as np
from yoloCompare import yoCompare
from mediaPipeCompare import mpCompare
import random
import time

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.75
font_thickness = 2
text_color = (0, 255, 0)  # Green (BGR format)
text_position = (50, 30)


class poseCompare:

    def __init__(self, modelName):
        self.model_name = modelName
        if modelName == 'yolo':
            self.model = yoCompare()
        else:
            self.model = mpCompare()

    def image_compare(self, img_path):
        img = cv2.imread(img_path)
        stream = cv2.VideoCapture(6)

        # Execute models
        streamSuccess = True
        while streamSuccess:
            streamSuccess, stream_image = stream.read()

            vid_frame, stream_frame = self.resize_images(img, stream_image)

            vid_sets = self.model.detect(vid_frame)
            stream_sets = self.model.detect(stream_frame)
            vid_annotated_image = self.model.draw_landmarks_on_image(vid_frame, vid_sets)
            stream_annotated_image = self.model.draw_landmarks_on_image(stream_frame, stream_sets)
            # Combine frames and add score text
            score = self.model.compare_detections(vid_sets, stream_sets)
            text = f'Score: {score[0]}'
            frame = np.hstack((vid_annotated_image, stream_annotated_image))
            cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            cv2.imshow('', frame)

            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
                break

    def pose_compare(self):
        vidObj = cv2.VideoCapture(6)
        success = True
        while success:
            success, image = vidObj.read()
            sets = self.model.detect(image)
            annotated_image = self.model.draw_landmarks_on_image(image, sets)
            if len(sets) == 2:
                score = self.model.compare_detections(sets[0], sets[1])
                text = f'Score: {score[0]:.2f}'
                cv2.putText(
                    annotated_image, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA
                )
            cv2.imshow('', annotated_image)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    def dance_compare(self, video_path):
        # Load video and start stream
        vid = cv2.VideoCapture(video_path)
        stream = cv2.VideoCapture(6)

        # Execute models
        vidSuccess = True
        streamSuccess = True

        # index in runningScore will corresond to a player. Players can steal score by swapping detections haha
        runningScore = {}
        colors = {}

        # setup output
        startTime = time.time()
        output_file = f'output/{startTime}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        frames = []

        while streamSuccess and vidSuccess:
            vidSuccess, vid_image = vid.read()
            streamSuccess, stream_image = stream.read()
            stream_image = cv2.flip(stream_image, 1)
            # Resize images to the same size
            vid_frame, stream_frame = self.resize_images(vid_image, stream_image)

            vid_sets = self.model.detect(vid_frame)
            stream_sets = self.model.detect(stream_frame)
            vid_annotated_image = self.model.draw_landmarks_on_image(vid_frame, vid_sets)
            stream_annotated_image = self.model.draw_landmarks_on_image(stream_frame, stream_sets)

            # Combine frames and add score text
            frame = np.hstack((vid_annotated_image, stream_annotated_image))

            # Check for existance of pose detections in both
            if len(vid_sets) < 1 or len(stream_sets) < 1:
                continue

            scores = self.model.compare_detections(vid_sets, stream_sets)
            for i, s in enumerate(scores):
                if i not in runningScore:
                    runningScore[i] = []
                    colors[i] = tuple(random.randint(0, 255) for _ in range(3))
                runningScore[i].append(s)
                rsAvg = round(sum(runningScore[i]) / len(runningScore[i]), 2) if runningScore[i] else 0
                text = f'Current Score: {f'{s:.2f}'}'
                text2 = f'Running Score: {rsAvg}'
                font_size = font_scale / len(runningScore)
                text_position = (50, 60 * i + 30)  # (x, y) coordinates (top-left corner)
                text2_position = (50, 60 * i + 60)
                cv2.putText(frame, text, text_position, font, font_size, colors[i], font_thickness, cv2.LINE_AA)
                cv2.putText(frame, text2, text2_position, font, font_size, colors[i], font_thickness, cv2.LINE_AA)

            frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
            frames.append(frame)
            cv2.imshow('', frame)

            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
                break

        # write video
        endTime = time.time()
        fps = len(frames) / (endTime - startTime)
        height = frames[0].shape[0]
        width = frames[0].shape[1]
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

    def resize_images(self, vid_image, stream_image):
        height = min(vid_image.shape[0], stream_image.shape[0])
        vid_frame = cv2.resize(vid_image, (int(vid_image.shape[1] * height / vid_image.shape[0]), height))
        stream_frame = cv2.resize(stream_image, (int(stream_image.shape[1] * height / stream_image.shape[0]), height))
        return vid_frame, stream_frame

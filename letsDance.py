import cv2
from yoloCompare import yoCompare
from mediaPipeCompare import mpCompare
import numpy as np


def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise Exception("Could not open video")
    return video


def dance_compare(video_path, model):
    vid = load_video(video_path)
    stream = cv2.VideoCapture(6)
    vidSuccess = True
    streamSuccess = True
    if model == "yolo":
        yolo = yoCompare()
        while vidSuccess and streamSuccess:    
            vidSuccess, vid_image = vid.read()
            streamSuccess, stream_image = stream.read()

            # Resize images to the same size
            if not vidSuccess or not streamSuccess:
                break
            vid_frame, stream_frame = resize_images(vid_image, stream_image)

            vid_result = yolo.detect(vid_frame)
            stream_result = yolo.detect(stream_frame)
            vid_annotated_image = yolo.annotate_image(vid_result)
            stream_annotated_image = yolo.annotate_image(stream_result)
            vid_keypoints = vid_result[0].keypoints.xy.cpu().numpy()
            stream_keypoints = stream_result[0].keypoints.xy.cpu().numpy()

            # if len(stream_keypoints) == 2:
            #     print(yolo.compare_detections(stream_keypoints[0], stream_keypoints[1]))

            frame = np.hstack((vid_annotated_image, stream_annotated_image))

            cv2.imshow("", frame)     
            
            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
                break

    if model == "mediapipe":
        mp = mpCompare()
        while streamSuccess:
            vidSuccess, vid_image = vid.read()
            streamSuccess, stream_image = stream.read()

            # Resize images to the same size
            if not vidSuccess or not streamSuccess:
                break
            vid_frame, stream_frame = resize_images(vid_image, stream_image)

            vid_sets = mp.detect(vid_frame)
            stream_sets = mp.detect(stream_image)
            vid_annotated_image = mp.draw_landmarks_on_image(vid_frame, vid_sets)
            stream_annotated_image = mp.draw_landmarks_on_image(stream_image, stream_sets)
            # if len(sets) == 2:
            #     print(mp.compare_detections(sets[0], sets[1]))

            frame = np.hstack((vid_annotated_image, stream_annotated_image))

            cv2.imshow('', frame)
            
            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
                break


def resize_images(vid_image, stream_image):
    height = min(vid_image.shape[0], stream_image.shape[0])
    vid_frame = cv2.resize(vid_image, (int(vid_image.shape[1] * height / vid_image.shape[0]), height))
    stream_frame = cv2.resize(stream_image, (int(stream_image.shape[1] * height / stream_image.shape[0]), height))
    return vid_frame, stream_frame

def main():
    dance_compare('dance_videos/dance1.mp4', 'yolo')
    # dance_compare('dance_videos/dance1.mp4', 'mediapipe')
    return 0


if __name__ == "__main__":
    main()
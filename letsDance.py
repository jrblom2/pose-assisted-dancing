import cv2
from yoloCompare import yoCompare
from mediaPipeCompare import mpCompare
import numpy as np


def load_image(img_path):
    return cv2.imread(img_path)


def main():
    # mp = mpCompare()
    yolo = yoCompare()
    yolo.dance_compare('dance_videos/dance1.mp4')
    # mp.dance_compare('dance_videos/dance1.mp4')
    # mp = mpCompare()
    # mp.pose_compare()
    return 0


if __name__ == "__main__":
    main()

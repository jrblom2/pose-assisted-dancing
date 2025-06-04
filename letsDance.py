from poseCompare import poseCompare


def main():
    pc = poseCompare('yolo')
    pc.dance_compare('dance_videos/ILoveCV.mp4', 'ilovecv.wav')
    # pc.image_compare('dance_videos/test.jpg')
    return 0


if __name__ == "__main__":
    main()

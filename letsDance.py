from poseCompare import poseCompare
import argparse


def main():
    parser = argparse.ArgumentParser(description='Pose comparison using YOLO or Mediapipe.')

    parser.add_argument(
        '--backend', choices=['yolo', 'mediapipe'], required=True, help='Pose detection backend to use.'
    )
    parser.add_argument(
        '--mode',
        choices=['dance', 'image'],
        required=True,
        help='Mode of operation: "dance" for video+audio, "image" for single image.',
    )
    parser.add_argument('--video', type=str, help='Path to the dance video file (for dance mode).')
    parser.add_argument('--audio', type=str, default=None, help='Optional path to the audio file (for dance mode).')
    parser.add_argument('--image', type=str, help='Path to the image file (for image mode).')
    parser.add_argument(
        '--stream', type=int, default=None, help='Optional video stream source (e.g. webcam index "0").'
    )

    args = parser.parse_args()

    pc = poseCompare(args.backend)

    if args.mode == 'dance':
        if not args.video:
            print("Error: --video is required for dance mode.")
            return
        pc.dance_compare(args.video, args.stream, args.audio)

    elif args.mode == 'image':
        if not args.image:
            print("Error: --image is required for image mode.")
            return
        pc.image_compare(args.image, args.stream)


if __name__ == "__main__":
    main()

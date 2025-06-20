# DANCEORDIE

This repository provides the code to compare multiple users to a reference dance video and score how similar the pose estimations are. Both yolo11 and mediapipe have been implemented to accomplish this, and multiple different examples ranging from side-by-side comparisions, image compariosns, and full dance comparisons are available with each framework.

The poses in this project are compared using a normalized, cosine similarity comparison.

## Authors
Grayson Snyder and Joseph Blom 

## Setup
To use the MediaPipe backend, you need to get the task in your local directory.
```
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

We recommend using a virtual environment for the projects dependancies.
```
python3 -m venv .venv
```

Install project dependancies after sourcing the virtual environment.
```
pip install -r requirements.txt
```

If you want audio for the video, run this command, subsituting the names accordingly for files.
```
ffmpeg -i your_dance_video.mp4 -q:a 0 -map a audio/your_dance_audio.wav
```

## Running
```
python letsDance.py --backend yolo --mode dance --video dance_videos/ILoveCV.mp4 --audio audio/ilovecv.wav --stream 0
```
| Argument    | Required | Description                                             |
| ----------- | -------- | ------------------------------------------------------- |
| `--backend` | Yes    | Pose detection backend: `yolo` or `mediapipe`.          |
| `--mode`    | Yes    | Operation mode: `dance` or `image`.                     |
| `--video`   | Yes | Path to a video file (required in `dance` mode).        |
| `--audio`   | Yes | Path to an audio file (required in `dance` mode).       |
| `--image`   | Yes  | Path to an image file (required in `image` mode).       |
| `--stream`  | No     | Optional video stream input (e.g. webcam index). |

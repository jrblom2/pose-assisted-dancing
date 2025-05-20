# DANCEORDIE

## Setup
Default video stream is set to 0, which is usually your webcam.

Need to get the task in your local directory
```
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

1.
```
python3 -m venv .venv
```

2.
```
pip install -r requirments.txt
```

## Running
```
python3 dance.py
```
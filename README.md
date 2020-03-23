# Sign language recognition with RNN and Mediapipe 
#### with multi hand tracking
Sign language gesture recognition using a reccurent neural network(RNN) with Mediapipe hand tracking. 

This project is for academic purpose. Thank you for Google's Mediapipe team :)

### Data Preprocessing with Mediapipe (Desktop on CPU)
Create training data on Desktop with input video using [Multi Hand Tracking](https://github.com/google/mediapipe/blob/master/mediapipe/docs/multi_hand_tracking_mobile_gpu.md).
Gesture recognition with deep learning model can be done with **hand landmark features** per frame with RNN training .

**CUSTOMIZE:**
- Use video input instead of Webcam on Desktop to train with video data
- Preprocess hand landmarks for every frame per one word and make it into one txt file

### 1. Set up Hand Tracking framework
* Install Medapipe
```shell
  git clone https://github.com/google/mediapipe.git
```
See the rest of installation documents [here](https://mediapipe.readthedocs.io/en/latest/install.html).
* Change **end_loop_calculator.h** file
```shell
  cd ~/mediapipe/mediapipe/calculators/core
  rm end_loop_calculator.h
```
to our new /end_loop_calculator.h file in the modified_mediapipe folder.

* Change **demo_run_graph_main.cc** file 
```shell
  cd ~/mediapipe/mediapipe/examples/desktop
  rm demo_run_graph_main.cc
```
to our new demo_run_graph_main.cc file in the modified_mediapipe folder.

* Change **landmarks_to_render_data_calculator.cc** file
```shell
  cd ~/mediapipe/mediapipe/calculators/util
  rm landmarks_to_render_data_calculator.cc
```
to our new landmarks_to_render_data_calculator.cc file in the modified_mediapipe folder.

### 2. Create your own training data
Make **train_videos** for each sign language word in one folder. Use **build.by** file to your mediapipe directory.
* Usage

To make mp4 file and txt file with mediapipe automatically, run
```shell
  python build.py --input_data_path=[INPUT_PATH] --output_data_path=[OUTPUT_PATH]
```
inside mediapipe directory.

IMPORTANT: Name the folder carefully as the folder name will be the label itself for the video data.
(DO NOT use space bar or '_' to your video name ex) Apple_pie (X))

For example:
```shell
input_video
├── Apple
│   ├── IMG_2733.MOV
│   ├── IMG_2734.MOV
│   ├── IMG_2735.MOV
│   └── IMG_2736.MOV
├── Bird
│   ├── IMG_2631.MOV
│   ├── IMG_2632.MOV
│   ├── IMG_2633.MOV
│   └── IMG_2634.MOV
└── Child
    ├── IMG_2472.MOV
    ├── IMG_2473.MOV
    ├── IMG_2474.MOV
    └── IMG_2475.MOV
    ...
```
The output path is initially an empty directory, and when the build is complete, Mp4 and txt folders are extracted to your folder path.

Created folder example:
```shell
output_data
├── Absolute
│   └── Apple
│       ├── IMG_2733.txt
│       ├── IMG_2734.txt
│       ├── IMG_2735.txt
│       └── IMG_2736.txt
|       ...
├── Relative
│   └── Apple
│       ├── IMG_2733.txt
│       ├── IMG_2734.txt
│       ├── IMG_2735.txt
│       └── IMG_2736.txt
│       ...
└── _Apple
     ├── IMG_2733.mp4 
     ├── IMG_2734.mp4
     ├── IMG_2735.mp4
     └── IMG_2736.mp4
```

Our ASL word example:
Your contribution is welcome [here](https://github.com/rabBit64/Sign-language-recognition-with-RNN-and-Mediapipe/tree/master/ASL)!
| word | word | word | word |
|:------:|:------:|:------:|:------:|
|Apple|Bird|Blue|Cents|
|Child|Cow|Drink|Green|
|Hello|Like|Metoo|No|
|Orange|Sorry|Thankyou|Where|
|Who|Yes|You|   |


### 3. Train RNN model

* Train
```shell
  python train.py --input_train_path=[INPUT_TRAIN_PATH] 
```
INPUT_TRAIN_PATH is the path to the output folder in the previous step. (either Relative or Absolute path) The model is saved as 'model.h5' in the current directory.

Watch [this video](https://www.youtube.com/watch?v=5epWNiv5EKk&t=77s) for the overall workflow.
[more details](https://www.slideshare.net/JiHyunKim204)






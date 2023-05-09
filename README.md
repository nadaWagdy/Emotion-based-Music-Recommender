# Emotion-based-Music-Recommender
A.I project that detects facial expressions and recommends music based on the detected emotions

**Needed installations**
- pip install deepface
- pip install opencv-python
- pip install numpy
- pip install keras
- pip install pillow
- pip install tensorflow

**The Emotion detection part is done in two parts**
- the first part is done using the DataSet - in forlder `Dataset` - and a model is trained on it
    - Find the model structure in `EmotionDetectionTrain.py` file 
    - Trained model weights is saved in `emotion_model.h5`
    - Test the model by uploading images in `TestEmotionDetector.py` file (test files can be found in `images` folder)

- the second part is done using deepface package in real time using camera
    - Find it in `videoEmotion.py` file

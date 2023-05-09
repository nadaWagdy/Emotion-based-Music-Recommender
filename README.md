# Emotion-based-Music-Recommender
A.I project that detects facial expressions and recommends music based on the detected emotions

**Needed installations**
- pip install deepface
- pip install opencv-python
- pip install numpy
- pip install keras
- pip install pillow
- pip install tensorflow

**The Emotion detection is done in two parts**
- the first part done using the DataSet - in forlder `Dataset` - and a model is trained on it
    - Find the model structure in `EmotionDetectionTrain.py` 
    - Trained model weights is saved in `emotion_model.h5`
    - Test the model by uploading images in `TestEmotionDetector.py` (test examples can be found in `images` folder)

- the second part done using deepface package in real time using camera
    - Find it in `videoEmotion.py`

import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json 
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights 
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# webcam
#cap = cv2.VideoCapture(0)


# download video from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("surprise.mp4")
img = cv2.imread("images/angryman.jpeg", cv2.IMREAD_COLOR)



face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in num_faces:
    cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
    roi_gray_frame = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    cv2.putText(img, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('Emotion Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
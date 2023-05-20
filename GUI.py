import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import cv2
from keras.models import model_from_json
import matplotlib.pyplot as plt
from tkinter import filedialog
from deepface import DeepFace

Emotion_Classes = ['Angry',
                  'Disgust',
                  'Fear',
                  'Happy',
                  'Neutral',
                  'Sad',
                  'Surprise']

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Music_Player = pd.read_csv("DataSet/data_moods.csv")
Music_Player = Music_Player[['name','artist','mood','popularity']]
Music_Player.head()


Music_Player["popularity"].value_counts()

Play = Music_Player[Music_Player['mood'] =='Calm' ]
Play = Play.sort_values(by="popularity", ascending=False)
Play = Play[:5].reset_index(drop=True)
# print(Play)

class App:
    def __init__(self, master):
        self.master = master
        master.title("Emotion Detection and Music Recommendation")

        frame = tk.Frame(master, bg='#d6edff')
        frame.pack(side='left', fill='y')

        self.label = tk.Label(frame, text="Music recommendations will be displayed here", bg='#d6edff')
        self.label.pack()

        self.canvas = tk.Canvas(master, width=800, height=480, bg='#e8e8e8')
        self.canvas.pack(side='right')

        self.button = tk.Button(frame, text="Select Image", command=self.select_image, bg='#00ced1', fg='white')
        self.button.pack(pady=10)

        self.img = None

        self.stopped = None

        self.live_emotion = None

        self.stream_button = tk.Button(frame, text="Stream from Camera", command=self.start_camera_stream, bg='#00ced1',
                                       fg='white')
        self.stream_button.pack(pady=10)

    def select_image(self):
        path = filedialog.askopenfilename(initialdir="/gui/images", title="Select a file",
                                                 filetypes=(("png files", "*.png"), ('Jpg Files', '*.jpg')))
        self.pred_and_plot(path)

    def load_and_prep_image(self, filename, img_shape=48):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)

        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(img, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

        return img, emotion_dict[maxindex]

    def Recommend_Songs(self, pred_class):
        if pred_class == 'Disgusted' or pred_class == 'disgust':
            Play = Music_Player[Music_Player['mood'] == 'Sad']
            Play = Play.sort_values(by="popularity", ascending=False)
            Play = Play[:15].reset_index(drop=True)

        if pred_class == 'Happy' or pred_class == 'Sad' or pred_class == 'sad' or pred_class == 'happy':
            Play = Music_Player[Music_Player['mood'] == 'Happy']
            Play = Play.sort_values(by="popularity", ascending=False)
            # Play = Play[:5].reset
            Play = Play[:15].reset_index(drop=True)

        if pred_class == 'Fearful' or pred_class == 'Angry' or pred_class == 'angry' or pred_class == 'fear':
            Play = Music_Player[Music_Player['mood'] == 'Calm']
            Play = Play.sort_values(by="popularity", ascending=False)
            Play = Play[:15].reset_index(drop=True)

        if pred_class == 'Surprised' or pred_class == 'Neutral' or pred_class == 'surprise' or pred_class == 'neutral':
            Play = Music_Player[Music_Player['mood'] == 'Energetic']
            Play = Play.sort_values(by="popularity", ascending=False)
            Play = Play[:15].reset_index(drop=True)

        recommendations = "\n\n".join([f"{i + 1}. {row['name']} - {row['artist']}" for i, row in Play.iterrows()])
        self.label.configure(text=recommendations)

    def pred_and_plot(self, filename):
        if self.img is not None:
            self.canvas.delete(self.img)

        img, pred_class = self.load_and_prep_image(filename)


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        width, height = img.size
        ratio = min(400 / width, 400 / height)
        img = img.resize((int(width*ratio), int(height*ratio)), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(img)

        self.canvas.create_image(410, 240, anchor='center', image=self.img)

        self.Recommend_Songs(pred_class)

        width, height = img.size
        ratio = min(400/width, 400/height)

        self.img = ImageTk.PhotoImage(img)

        self.canvas.create_image(410, 240, anchor='center', image=self.img)

        self.Recommend_Songs(pred_class)

    def start_camera_stream(self):
        self.camera_window = tk.Toplevel(self.master)
        self.camera_window.title("Camera Stream")

        self.video_label = tk.Label(self.camera_window)
        self.video_label.pack()

        self.stop_button = tk.Button(self.camera_window, text="Stop Stream", command=self.stop_camera_stream)
        self.stop_button.pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        self.show_frame()

        self.stopped = False

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,
                        result[0]['dominant_emotion'],
                        (50, 50),
                        font, 3,
                        (0, 0, 255),
                        2,
                        cv2.LINE_4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((640, 480), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(img)
            self.video_label.config(image=self.img)
            self.live_emotion = result[0]['dominant_emotion']
        if not self.stopped:
            self.video_label.after(10, self.show_frame)

    def stop_camera_stream(self):
        self.stopped = True
        print(self.live_emotion)
        self.Recommend_Songs(self.live_emotion)
        self.camera_window.destroy()
        self.cap.release()


root = tk.Tk()
app = App(root)
root.mainloop()

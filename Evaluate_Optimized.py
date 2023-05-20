import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import tensorflow as tf

Emotion_Classes = ['Angry',
                  'Fear',
                  'Happy',
                  'Neutral',
                  'Sad',
                  'Surprise']

# load json and create model
json_file = open('cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("cnn_model.h5")
print("Loaded model from disk")

validation_data = ImageDataGenerator(rescale=1./255)

test_generator = validation_data.flow_from_directory(
    'DataSet2/Testing/Testing',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)
test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
predictions = emotion_model.predict_generator(test_generator, steps=test_steps_per_epoch)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(true_classes, predicted_classes)

print(cm)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Emotion_Classes)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

print("-----------------------------------------------------------------")
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))


print("-----------------------------------------------------------------")
overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print("Overall Accuracy:", overall_accuracy)


Random_batch = np.random.randint(0, len(test_generator) - 1)

Random_Img_Index = np.random.randint(0, 64 - 1 , 10)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):

    Random_Img = test_generator[Random_batch][0][Random_Img_Index[i]]

    Random_Img_Label = np.argmax(test_generator[Random_batch][1][Random_Img_Index[i]])

    Model_Prediction = np.argmax(emotion_model.predict( tf.expand_dims(Random_Img, axis=0) , verbose=0))

    ax.imshow(Random_Img)

    if Emotion_Classes[Random_Img_Label] == Emotion_Classes[Model_Prediction]:
        color = "green"
    else:
        color = "red"
    ax.set_title(f"True: {Emotion_Classes[Random_Img_Label]}\nPredicted: {Emotion_Classes[Model_Prediction]}",
                 color=color)
plt.show()
plt.tight_layout()

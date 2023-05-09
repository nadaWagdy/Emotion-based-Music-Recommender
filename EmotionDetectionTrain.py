from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

training_data = ImageDataGenerator(rescale=1./255)
validation_data = ImageDataGenerator(rescale=1./255)

# preprocess data (images) train
training_generator = training_data.flow_from_directory(
    'DataSet/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# preprocess data (images) test
validation_generator = validation_data.flow_from_directory(
    'DataSet/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax')) # the seven categories which are the seven emotions

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy']) # with learning rate of 0.0001

# train CNN (model)
emotion_model_info = emotion_model.fit_generator(
    training_generator,
    steps_per_epoch=28709 // 64, # total number of images is 28709
    epochs=50,
    validation_data= validation_generator,
    validation_steps=7178 // 64 # total number of validation images is 7178
)

# save model structure to json file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weights in .h5 file
emotion_model.save_weights('emotion_model.h5')
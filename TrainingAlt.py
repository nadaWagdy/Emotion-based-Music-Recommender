from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

training_data = ImageDataGenerator(rescale=1./255)
validation_data = ImageDataGenerator(rescale=1./255)

# preprocess data (images) train
training_generator = training_data.flow_from_directory(
    'DataSet2/Training/Training',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# preprocess data (images) test
validation_generator = validation_data.flow_from_directory(
    'DataSet2/Testing/Testing',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# create model structure
cnn_model = Sequential()

cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))


cnn_model.add(Flatten())
cnn_model.add(Dense(1024, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(6, activation='softmax')) # the seven categories which are the seven emotions

cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy']) # with learning rate of 0.0001

# train CNN (model)
cnn_model_info = cnn_model.fit_generator(
    training_generator,
    steps_per_epoch=28273 // 64, # total number of images is 28709
    epochs= 10,
    validation_data= validation_generator,
    validation_steps=7067 // 64, # total number of validation images is 7178
    callbacks=[early_stop]
)

# save model structure to json file
model_json = cnn_model.to_json()
with open("cnn_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weights in .h5 file
cnn_model.save_weights('cnn_model.h5')

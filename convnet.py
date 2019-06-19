from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np

img_width, img_height = 255, 255

train_data_dir = 'data/train/'
validation_data_dir = 'data/validation/'
nb_train_samples = 168
nb_validation_samples = 168
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

untitled = image.img_to_array(image.load_img('data/test/untitled/untitled1.jpeg',
                                             target_size=(255, 255)))
untitled = np.expand_dims(untitled, axis=0)

images = np.vstack([untitled])
classes = model.predict_classes(images, batch_size=10)
print("Predicted class using untitled is:", classes)

tam = image.img_to_array(image.load_img('data/test/tam/tam1.jpeg',
                                        target_size=(255, 255)))
tam = np.expand_dims(tam, axis=0)

images = np.vstack([tam])
classes = model.predict_classes(images, batch_size=10)
print("Predicted class using tam is:", classes)

united_airline = image.img_to_array(image.load_img('data/validation/united_airline/united_airline1.jpeg',
                                             target_size=(255, 255)))
united_airline = np.expand_dims(united_airline, axis=0)

images = np.vstack([united_airline])
classes = model.predict_classes(images, batch_size=10)
print("Predicted class using united_airline is:", classes)

delta = image.img_to_array(image.load_img('data/validation/delta/delta1.jpeg',
                                             target_size=(255, 255)))
delta = np.expand_dims(delta, axis=0)

images = np.vstack([delta])
classes = model.predict_classes(images, batch_size=10)
print("Predicted class using delta is:", classes)

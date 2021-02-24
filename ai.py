import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=.2, zoom_range=.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("./dataset/training_set", target_size=(64,64), batch_size=32, class_mode='binary')
test_set = train_datagen.flow_from_directory("./dataset/test_set", target_size=(64,64), batch_size=32, class_mode='binary')

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=25, verbose=1, restore_best_weights=True)

model = Sequential()

model.add(Conv2D(filters = 1024, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))
model.add(MaxPool2D(pool_size = 3, strides = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 1024, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(MaxPool2D(pool_size = 3, strides = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 1024, kernel_size = 5, activation = 'relu', padding = 'same'))
model.add(MaxPool2D(pool_size = 5, strides = 2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units = 256, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x = training_set, validation_data = test_set, batch_size = 16, epochs = 50, callbacks=[es])
model.summary()

images = []
name = []
folder_path = './test_preditions'
for img in os.listdir(folder_path):
    name.append(os.path.join(folder_path, img))
    img = os.path.join(folder_path, img)
    img = image.load_img(img, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

images = np.vstack(images)
classes = model.predict_classes(images, batch_size=10)
c = 0
for i in classes:
    if i == 1:
        print(name[c], 'dog')
    else:
        print(name[c], 'cat')
    c+=1

test_image = image.load_img('./test_preditions/cat3.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction) 

model.save('m.h5')
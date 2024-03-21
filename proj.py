
# Importing necessary modules.
import tensorflow as tf
from tensorflow.keras.utils import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
# %matplotlib inline

BASE_DIR = '/Users/vinayak/Desktop/project/archive-2 2/utkface_aligned_cropped/UTKFace'
age_labels = []
gender_labels = []
image_paths = []




#image_filenames = os.listdir(BASE_DIR)
#random.shuffle(image_filenames)

#for image in tqdm(image_filenames):
#image_path = os.path.join(BASE_DIR, image)
#img_components = image.split('_')
#age_label = int(img_components[0])
#gender_label = int(img_components[1])

# Append the image_path, age_label, and gender_label
#age_labels.append(age_label)
#gender_labels.append(gender_label)
#image_paths.append(image_path)



# print(f'Number of age_labels: {len(age_labels)}, Number of gender_labels: {len(gender_labels)}, Number of image_paths: {len(image_paths)}')
# Iterate through image files in BASE_DIR
for image in tqdm(os.listdir(BASE_DIR)):
    # Ignore hidden files and directories
    if not image.startswith('.'):
        image_path = os.path.join(BASE_DIR, image)
        img_components = image.split('_')
        
        # Ensure img_components has at least two elements to prevent index out of range error
        if len(img_components) >= 2:
            try:
                # Extract age and gender labels from image file name
                age_label = int(img_components[0])
                gender_label = int(img_components[1])

                # Append the image_path, age_label, and gender_label
                age_labels.append(age_label)
                gender_labels.append(gender_label)
                image_paths.append(image_path)
            except ValueError:
                print(f"Invalid age or gender label in file name: {image}. Skipping...")
        else:
            print(f"Invalid file name format: {image}. Skipping...")

print(f'Number of age_labels: {len(age_labels)}')

print(f'Number of age_labels: {len(age_labels)}')
gender_mapping = {
    1: 'Female',
    0: 'Male'
}

import pandas as pd
df = pd.DataFrame()
df['image_path'], df['age'], df['gender'] = image_paths, age_labels, gender_labels
df.head(5)

from PIL import Image

rand_index = random.randint(0, len(image_paths))
age = df['age'][rand_index]
gender = df['gender'][rand_index]
IMG = Image.open(df['image_path'][rand_index])
# plt.title(f'Age: {age} Gender: {gender_mapping[gender]}')
# plt.axis('off')
# plt.imshow(IMG)

# Age distribution
sns.distplot(df['age'])

sns.countplot(df['gender'])

# plt.figure(figsize=(20, 20))
samples = df.iloc[0:16]

# for index, sample, age, gender in samples.itertuples():
#     plt.subplot(4, 4, index + 1)
#     img = load_img(sample)
#     img = np.array(img)
#     plt.axis('off')
#     plt.title(f'Age: {age} Gender: {gender_mapping[gender]}')
#     plt.imshow(img)

"""Feature Extraction"""

def extract_image_features(images):
    features = list()

    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

X = extract_image_features(df['image_path'])

X.shape

X = X / 255.0

y_gender = np.array(df['gender'])
y_age = np.array(df['age'])

input_shape = (128, 128, 1)

#inputs = Input((input_shape))
#conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
#max_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
#conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(max_1)
#max_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
#conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(max_2)
#max_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
#conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(max_3)
#max_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

#conv_5 = Conv2D(512, kernel_size=(3, 3), activation='relu')(max_4)
#max_5 = MaxPooling2D(pool_size=(2, 2))(conv_5)
#conv_6 = Conv2D(1024, kernel_size=(3, 3), activation='relu')(max_5)
#max_6 = MaxPooling2D(pool_size=(2, 2))(conv_6)
#conv_7 = Conv2D(2048, kernel_size=(3, 3), activation='relu')(max_6)
#max_7 = MaxPooling2D(pool_size=(2, 2))(conv_7)
#conv_8 = Conv2D(4096, kernel_size=(3, 3), activation='relu')(max_7)
#max_8 = MaxPooling2D(pool_size=(2, 2))(conv_8)
#conv_9 = Conv2D(8192, kernel_size=(3, 3), activation='relu')(max_8)
#max_9 = MaxPooling2D(pool_size=(2, 2))(conv_9)
#conv_10 = Conv2D(16384, kernel_size=(3, 3), activation='relu')(max_9)
#max_10 = MaxPooling2D(pool_size=(2, 2))(conv_10)

inputs = Input((input_shape))
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
max_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(max_1)
max_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(max_2)
max_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(max_3)
max_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

conv_5 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(max_4)
max_5 = MaxPooling2D(pool_size=(2, 2))(conv_5)
conv_6 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(max_5)
max_6 = MaxPooling2D(pool_size=(2, 2))(conv_6)
conv_7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(max_6)
max_7 = MaxPooling2D(pool_size=(2, 2))(conv_7)


flatten = Flatten()(max_4)

# fully connected layers
dense_1 = Dense(256, activation='relu')(flatten)
dense_2 = Dense(256, activation='relu')(flatten)

dense_3 = Dense(256, activation='relu')(flatten)
dense_4 = Dense(256, activation='relu')(flatten)
dense_5 = Dense(256, activation='relu')(flatten)


dropout_1 = Dropout(0.3)(dense_1)
dropout_2 = Dropout(0.3)(dense_2)

dropout_3 = Dropout(0.3)(dense_3)
dropout_4 = Dropout(0.3)(dense_4)
dropout_5 = Dropout(0.3)(dense_5)


output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
output_2 = Dense(1, activation='relu', name='age_out')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'],
              optimizer='adam', metrics=['accuracy'])

# plot the model
from tensorflow.keras.utils import plot_model
plot_model(model)

history = model.fit(x=X, y=[y_gender, y_age],
                    batch_size=32, epochs=1, validation_split=0.2)

"""Plot Results"""

# plot results for gender
acc = history.history['gender_out_accuracy']
val_acc = history.history['val_gender_out_accuracy']
epochs = range(len(acc))

# plt.plot(epochs, acc, 'b', label='Training Accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
# plt.title('Accuracy Graph')
# plt.legend()
# plt.figure()

loss = history.history['gender_out_loss']
val_loss = history.history['val_gender_out_loss']

# plt.plot(epochs, loss, 'b', label='Training Loss')
# plt.plot(epochs, val_loss, 'r', label='Validation Loss')
# plt.title('Loss Graph')
# plt.legend()
# # plt.show()

# plot results for age
loss = history.history['age_out_loss']
val_loss = history.history['val_age_out_loss']
epochs = range(len(loss))

# plt.plot(epochs, loss, 'b', label='Training Loss')
# plt.plot(epochs, val_loss, 'r', label='Validation Loss')
# plt.title('Loss Graph')
# plt.legend()
# # plt.show()

"""Predicting Test Data"""

def get_image_features(image):
  img = load_img(image, grayscale=True)
  img = img.resize((128, 128), Image.ANTIALIAS)
  img = np.array(img)
  img = img.reshape(1, 128, 128, 1)
  img = img / 255.0
  return img

img_to_test = '/Users/vinayak/Downloads/IMG_C7645AA5CB93-1.jpeg'
features = get_image_features(img_to_test)
pred = model.predict(features)
gender = gender_mapping[round(pred[0][0][0])]
age = round(pred[1][0][0])

plt.title(f'Predicted Age: {age} Predicted Gender: {gender}')
plt.axis('off')
plt.imshow(np.array(load_img(img_to_test)))
plt.show()
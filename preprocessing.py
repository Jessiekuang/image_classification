
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def get_y(data, label, enc, val=False):
    # One-Hot Encoding
    if val:
        y = pd.DataFrame(enc.transform(data[[label]]).toarray())
    else:
        y = pd.DataFrame(enc.fit_transform(data[[label]]).toarray())
    return y


def load_image(dataframe):
    x_image = []
    for index, row in dataframe.iterrows():
        image_id = str(row['id'])
        img = load_img("./data/suffled-images/shuffled-images/"+ image_id + ".jpg")
        img = img_to_array(img)
        x_image.append(img)
    # normalize the data
    x_image = np.array(x_image)
    # normalize the data
    x_image.astype('float32')
    x_image /= 255
    return x_image


def process_data(data, enc, val=False):
    y = get_y(data, "category", enc, val)
    x = load_image(data)
    return x, y

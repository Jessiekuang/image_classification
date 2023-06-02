
from keras.models import Sequential, Model, load_model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling1D, MaxPooling1D, \
    BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Add, Input, Embedding, Lambda, concatenate, Conv1D, GRU


import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def Image_model():
    model = Sequential()
    # model.add(Input(shape=x_train_img.shape[1:], dtype='float32'))
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(80, 60, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(96, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(96, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(27, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


im = Image_model()
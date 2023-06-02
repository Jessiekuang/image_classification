import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling1D, MaxPooling1D, \
    BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Add, Input, Embedding, Lambda, concatenate, Conv1D, GRU
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import optimizers

import math
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

from tkinter import *
import tkinter.filedialog
import cv2
from PIL import Image
import pyttsx3

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from model import *
from preprocessing import *

# parameters for this script
batch_size = 50
num_classes = 27


def train(model, train_size, val_size, t_generator, v_generator):
    # compile model
    # RMSprop/'adagrad'/'adam'
    opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)

    # Compile the model before using it
    def compile_func(model, opt=opt):
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return model

    model = compile_func(model)
    steps_per_epoch = int(np.ceil(train_size/float(batch_size)))
    model.fit_generator(t_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=val_size,
                        epochs=40,
                        validation_data=v_generator)
    model.save('./my_model')
    return model


if __name__ == "__main__":
    # read text -> y_train_text, y_validation_text, train_text_data, validate_text_data
    train_text = pd.read_csv("./data/train.csv")
    train_text_data, validate_text_data = train_test_split(train_text, test_size=0.10)

    enc = OneHotEncoder(handle_unknown='ignore')
    x_train_img, y_train = process_data(train_text_data, enc, val=False)
    x_validate_img, y_validate = process_data(train_text_data, enc, val=True)

    generator = ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False)

    t_generator = generator.flow(x_train_img, y=y_train, batch_size=32)
    v_generator = generator.flow(x_validate_img, y=y_validate, batch_size=32)

    # I_model
    i_model = Image_model()
    i_model = train(i_model, x_train_img.shape[0], x_validate_img.shape[0], t_generator, v_generator)
import pathlib
import random
import numpy as np
import keras
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import *
from SharedParams import *

num_hidden = 100
# load datasets:
X = np.load('data/X_0.npy')
y = np.load('data/y_0.npy')
num_size = X.shape[0]
num_classes = len(get_amino_3grams_dict()) + 1

inputs = [Input(shape=(num_size, num_classes), name='input_{}'.format(i)) for i in range(get_context_size())]
dense_layers = [Dense(num_hidden, activation='relu')(inp) for inp in inputs]
hidden = Average()(dense_layers)
output = Dense(num_classes, activation='softmax')(hidden)

model = Model(inputs, output)
optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit({'input_{}'.format(i): keras.utils.to_categorical(X[:, i], num_classes=num_classes) for i in range(get_context_size())},
          keras.utils.to_categorical(y, num_classes=num_classes), batch_size=1, epochs=1, validation_split=.2, shuffle=True)

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
X = np.load('data/X_0.npy')[450000:500000, :]
y = np.load('data/y_0.npy')[450000:500000]
num_size = X.shape[0]
num_classes = len(get_amino_3grams_dict()) + 1

inputs = [Input(shape=(num_classes,), name='input_{}'.format(i)) for i in range(get_context_size())]
dense_layers = [Dense(num_hidden, activation='relu')(inp) for inp in inputs]
dropout_layers = [Dropout(rate=.5)(dense) for dense in dense_layers]
hidden = Average()(dropout_layers)
dropout_hidden = Dropout(rate=.5)(hidden)
output = Dense(num_classes, activation='softmax')(dropout_hidden)

model = Model(inputs, output)
optimizer = Adam(learning_rate=.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

for i in range(20):
    num_epochs = 10
    model.fit({'input_{}'.format(i): keras.utils.to_categorical(X[:, i].reshape((num_size, 1)), num_classes=num_classes) for i in range(get_context_size())},
              keras.utils.to_categorical(y.reshape((num_size, 1)), num_classes=num_classes), batch_size=32, epochs=num_epochs, validation_split=.2, shuffle=True)
    model.save('models/model1_{}.h5'.format(num_epochs * (i + 1)))
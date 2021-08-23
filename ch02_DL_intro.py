# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 07:49:24 2021

@author: user
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# %% 

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# %% 

NUM_CLASSES = 10

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

y_train = keras.utils.to_categorical( y_train, NUM_CLASSES )
y_test = keras.utils.to_categorical( y_test, NUM_CLASSES )

# %% sequential approach

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(32,32,3)),
    keras.layers.Flatten(),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
'''
model = keras.models.Sequential([
    keras.layers.Dense(200, activation='relu', input_shape=(32,32,3)),
    keras.layers.Flatten(),
    keras.layers.Dense(150, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
'''

model.summary()

# %% functional approach

input_layer = keras.layers.Input(shape=(32,32,3))

x = keras.layers.Flatten()(input_layer)
x = keras.layers.Dense(200, activation='relu')(x)
x = keras.layers.Dense(150, activation='relu')(x)

output_layer = keras.layers.Dense(10, activation='softmax')(x)

model = keras.models.Model(input_layer, output_layer)

model.summary()

# %% 

opt = keras.optimizers.Adam(lr=0.0005)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# %% 

model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

# %% 

model.evaluate(x_test, y_test)

# %% 

CLASSES = np.array(['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 
                    'horse', 'ship', 'truck'])

preds = model.predict(x_test)

preds_single = CLASSES[ np.argmax( preds, axis=-1 ) ]
actual_single = CLASSES[ np.argmax( y_test, axis=-1 ) ]

# %% 

import matplotlib.pyplot as plt

n_to_show = 10
indeces = np.random.choice( len(x_test), n_to_show )

fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indeces):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred: ' + str(preds_single[idx]), fontsize=10, 
            ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'actu: ' + str(actual_single[idx]), fontsize=10, 
            ha='center', transform=ax.transAxes)
    ax.imshow(img)


# %% 

input_layer = keras.layers.Input(shape=(32,32,3))

conv1 = keras.layers.Conv2D(filters=10, kernel_size=(4,4), strides=2,
                            padding='same')(input_layer)
conv2 = keras.layers.Conv2D(filters=20, kernel_size=(3,3), strides=2,
                            padding='same')(conv1)
flat = keras.layers.Flatten()(conv2)

output_layer = keras.layers.Dense(units=10, activation='softmax')(flat)

model = keras.models.Model(input_layer, output_layer)

# %% 

opt = keras.optimizers.Adam(lr=0.0005)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# %% 

model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

# %% 

model.evaluate(x_test, y_test)

# %% 

CLASSES = np.array(['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 
                    'horse', 'ship', 'truck'])

preds = model.predict(x_test)

preds_single = CLASSES[ np.argmax( preds, axis=-1 ) ]
actual_single = CLASSES[ np.argmax( y_test, axis=-1 ) ]

# %% 

n_to_show = 10
indeces = np.random.choice( len(x_test), n_to_show )

fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indeces):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred: ' + str(preds_single[idx]), fontsize=10, 
            ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'actu: ' + str(actual_single[idx]), fontsize=10, 
            ha='center', transform=ax.transAxes)
    ax.imshow(img)

# %% 

input_layer = keras.layers.Input(shape=(32,32,3))

x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                            padding='same')(input_layer)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU()(x)

x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2,
                            padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU()(x)

x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                            padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU()(x)

x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2,
                            padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU()(x)

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(128)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU()(x)
x = keras.layers.Dropout(rate=0.5)(x)

x = keras.layers.Dense(NUM_CLASSES)(x)

output_layer = keras.layers.Activation('softmax')(x)

model = keras.models.Model(input_layer, output_layer)

model.summary()

# %% 

opt = keras.optimizers.Adam(lr=0.0005)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# %% 

model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

# %% 

model.evaluate(x_test, y_test)

# %% 

CLASSES = np.array(['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 
                    'horse', 'ship', 'truck'])

preds = model.predict(x_test)

preds_single = CLASSES[ np.argmax( preds, axis=-1 ) ]
actual_single = CLASSES[ np.argmax( y_test, axis=-1 ) ]

# %% 

n_to_show = 10
indeces = np.random.choice( len(x_test), n_to_show )

fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indeces):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred: ' + str(preds_single[idx]), fontsize=10, 
            ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'actu: ' + str(actual_single[idx]), fontsize=10, 
            ha='center', transform=ax.transAxes)
    ax.imshow(img)





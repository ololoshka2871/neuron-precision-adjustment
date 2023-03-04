#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import Sequential, layers, optimizers

c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

model = Sequential()
model.add(layers.Dense(units=1, input_shape=(1,), activation='linear'))
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.1))

log = model.fit(c,f, epochs=500, verbose=0) # type: ignore

print("Обучение завершено")

print("Predict {} *C -> {} *F".format(100, model.predict([100])[0]))
print(f"Weigths: {model.get_weights()}")

plt.plot(log.history['loss'])
plt.show()
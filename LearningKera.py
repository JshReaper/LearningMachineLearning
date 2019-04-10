import tensorflow as tf
import pandas as p
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test, y_test) = mnist.load_data()

data = p.DataFrame(x_train[0])
data

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

data = p.DataFrame(x_train[0])
data

plt.imshow(x_train[24],cmap=plt.cm.binary)

print(y_train[24])
1

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)

val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss)
print(val_acc)

predictions = model.predict(x_test)

import numpy as np

print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()
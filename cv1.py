import  cv2
import tensorflow as tf
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import image
import numpy as np
from sklearn.metrics.scorer import metric

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')

x_train /= 255

#Chuyển y_train về dạng one host vecto
y_train = np_utils.to_categorical(y_train, 10)

# Mang CNN
img_width, img_height = 28, 28
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
print(input_shape)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3), input_shape=input_shape, activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(500, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

#Cấu hình loss
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
# model.summary()


#Train model
#truyền vào 1 ảnh có kích thước (28,28,1) nhưng khi train cần truyền vào nhiều ảnh nên cần mảng gồm 4 phần tử
# [1000, 28, 28, 1]
model.fit(x_train, y_train,epochs=2)
model.save('model.h5')
img = image.load_img('123.png', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
print(images)

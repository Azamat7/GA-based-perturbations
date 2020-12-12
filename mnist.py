import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import PIL
import numpy

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
train_labels = train_labels.reshape((train_labels.shape[0], 1))
test_labels = test_labels.reshape((test_labels.shape[0], 1))

train_images, test_images = train_images / 255.0, test_images / 255.0
im = train_images[1].reshape((28,28))
print(im.shape)
print(train_labels[1])
im = PIL.Image.fromarray((im*255.0).astype(numpy.uint8))
im.save("mnist.png")

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
# model.add(layers.Dense(10, activation='softmax'))

# model.summary()

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))

# model.save('mnist')
# model.save_weights('mnist_w')

# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print('model, accuracy: {:5.2f}%'.format(100 * acc))

model = tf.keras.models.load_model('mnist')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
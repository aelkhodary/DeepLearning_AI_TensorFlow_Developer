'''
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=I-haLncrva5L
##Exercise 2:
Let's now look at the layers in your model. Experiment with different values for the dense layer with 512 neurons.
What different results do you get for loss, training time etc? Why do you think that's the case?
'''

import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
#https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
print(type(classifications))
print(classifications.size)
print(classifications[0])
print(test_labels[0])

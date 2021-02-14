'''
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=I-haLncrva5L
'''

import tensorflow as tf
from tensorflow import keras

fashion_minst = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_minst.load_data()
print(tf.__version__)
'''
train_object = fashion_minst.load_data()

print('----------------------- train_data -------------------------------')
print(type(train_object[0][0]))
print(train_object[0][0])

print(type(train_object[0][1]))
print(train_object[0][1])
print('----------------------- test_data -------------------------------')
print(type(train_object[1][0]))
print(train_object[1][0])

print(type(train_object[1][1]))
print(train_object[1][1])

'''

'''

Remember last time we had a sequential with just one layer in it.
Now we have three layers.
The important things to look at are the first and the last layers.
The last layer has 10 neurons in it because we have ten classes of clothing in the dataset.
They should always match.


The first layer is a flatten layer with the input shaping 28 by 28. Now,
if you remember our images are 28 by 28,
so we're specifying that this is the shape that we should expect the data to be in.
Flatten takes this 28 by 28 square and turns it into a simple linear array.

The interesting stuff happens in the middle layer, sometimes also called a hidden layer.
This is a 128 neurons in it, and I'd like you to think about these as variables in a function.
Maybe call them x1, x2 x3, etc.
Now,there exists a rule that incorporates all of these that turns the 784 (28*28)
values of an ankle boot into the value nine,
and similar for all of the other 70,000.

It's too complex a function for you to see by mapping the images yourself,
 but that's what a neural net does. So,
 for example, if you then say the function was y equals w1 times x1, plus w2 times x2, plus w3 times x3,
 all the way up to a w128 times x128. By figuring out the values of w, then y will be nine,

 w0x0+w1x1+w2x2....wNxN=9
'''
import numpy as np
np.set_printoptions(linewidth=200)

'''
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
'''

print(train_labels[1000])
print(train_images[1000])


train_images  = train_images / 255.0
test_images = test_images / 255.0

print(train_labels[1000])
print(train_images[1000])

'''
Let's now design the model.
There's quite a few new concepts here, but don't worry, you'll get the hang of them.
'''

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

'''

**Sequential**: That defines a SEQUENCE of layers in the neural network

**Flatten**: Remember earlier where our images were a square, when you printed them out?
         Flatten just takes that square and turns it into a 1 dimensional set.

**Dense**: Adds a layer of neurons
     Each layer of neurons need an activation function to tell them what to do.
     There's lots of options, but just use these for now.

**Relu** effectively means "If X>0 return X, else return 0" --
         so what it does it it only passes values 0 or greater to the next layer in the network.

**Softmax** takes a set of values, and effectively picks the biggest one, so,
            for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05],
            it saves you from fishing through it looking for the biggest value
            , and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!

'''

'''

The next thing to do, now the model is defined, is to actually build it.
You do this by compiling it with an optimizer and loss function as before -- and then you train it by calling **model.
fit ** asking it to fit your training data to your training labels -- i.e.
have it figure out the relationship between the training data and its actual labels,
so in future if you have data that looks like the training data,
then it can make a prediction for what that data would look like.

'''

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)


'''
Once it's done training -- you should see an accuracy value at the end of the final epoch. It might look something like 0.9098.
 This tells you that your neural network is about 91% accurate in classifying the training data. I.E.,
 it figured out a pattern match between the image and the labels that worked 91% of the time.
 Not great, but not bad considering it was only trained for 5 epochs and done quite quickly.

But how would it work with unseen data? That's why we have the test images. We can call model.evaluate, and pass in the two sets,
and it will report back the loss for each. Let's give it a try:

'''

print(model.evaluate(test_images, test_labels))

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

'''
What does this list represent?
It's the probability that this item is each of the 10 classes


How do you know that this list tells you that the item is an ankle boot?
he 10th element on the list is the biggest, and the ankle boot is labelled 9
Both the list and the labels are 0 based, so the ankle boot having label 9 means that it is the 10th of the 10 classes.
The list having the 10th element being the highest value means that the Neural Network has predicted that the item it is classifying is most likely an ankle boot

'''

# Convolutional Neural Network

# Part 1 - Building the CNN

"""
Preparing the data:
    -Separate your images into 2 folders, training and test sets
    -Inside each of those folders, have separate folders for cats and dogs
    -That's how Keras will know the proper labels (ground true y)
You still use about 80% for training

Feature Scaling will come later
"""


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
"""32 filters, 3x3 filters, image size (h, w, c), relu activation
padding default is "same", stride default is 1
Since our images are all different sizes, we will force them into 64x64 
before we fit the classifier to the data.
When tuning, you can modify the image size, change number and size of filters,
and add dropout between layers"""
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
"""2x2 filter, strides default to filter size
we no longer need to set the input_shape, since the model knows
the shape from the previous step"""
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding dropout
classifier.add(Dropout(0.1))

"""Adding a second convolutional layer after training with only one"""
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding dropout
classifier.add(Dropout(0.1))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
"""128 nodes (common practice to choose a power of 2), relu"""
classifier.add(Dense(units = 128, activation = 'relu'))

# Adding dropout
classifier.add(Dropout(0.05))

"""output layer (binary- cat or dog), 1 node, sigmoid"""
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN to the images
"""Image augmentation to reduce overfitting,
since we only have 8k/2k images in the training/test sets.
Keras will transform our images in random ways to create new images.
This includes scaling (see rescale)
more info: https://keras.io/preprocessing/image/"""

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.4,
                                   zoom_range = 0.4,
                                   rotation_range = 45,
                                   width_shift_range = 0.4,
                                   height_shift_range = 0.4,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


"""Now create the training/test sets. This includes resizing (see target_size).
The target size can be increased to give higher res pictures to the model for 
improved accuracy.
You can change the directory to whatever images you want to train/test on.
This tutorial uses cats and dogs. 
Inside of each of the folders "training_set" and "test_set", 
there are two folders called "cats" and "dogs" """

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), #size that the model expects
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

"""This will take 1.5 - 3.5  mins per epoch on my machine"""
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32,  # number of images in training set div by batch size
                         epochs = 1,               #normally this would be higher with more CPU/GPU power
                         validation_data = test_set,
                         validation_steps = 2000/32)  # number of images in test set div by batch size


"""Saving/loading whole models (architecture + weights + optimizer state):
https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
"""
# To save a trained model
classifier.save('cnn_trained.h5')

# To load the model after the kernal has been restarted:
from keras.models import load_model
classifier = load_model('cnn_trained.h5')

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image

# Load the image you want to make a prediction on
test_image = image.load_img('dataset/single_prediction/cat_or_dog_4.jpg', target_size = (64, 64))

# Turn into a 3 dimensional array for RGB channel
test_image = image.img_to_array(test_image)

# Add a 4th dimension bc the predict method expects 4 dimensions
"""The 4th dim corresponds to the batch, bc predict cannot take a single 
input by itself. It must be in a batch, even if it's a batch of 1"""
test_image = np.expand_dims(test_image, axis = 0)

# Get prediction
result = classifier.predict(test_image)

# this will tell us the mapping bt cats/dogs and their associated values (1 or 0)
training_set.class_indices  #output: {'cats': 0, 'dogs': 1}

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

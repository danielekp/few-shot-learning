# few-shot-learning

Neural network prediciton with few examples.

_____

The Omniglot dataset is used for training ( a collection of 19280 images of 964 characters from 30 alphabets. There are 20 images for each of the 964 characters in the dataset.)

We follow the ***meta-learning approach***, i.e. the network learn how to learn new tasks using multiple training examples of tasks.

The meta-learning is performed using ***episodic training***; in each episode one training batch is processed.

The ***embedding CNN*** is a convolutional neural network whose aim is to embed images into a lower-dimensional space.

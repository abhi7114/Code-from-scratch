# Code-from-scratch : Neural Network

This Python code implements a basic neural network architecture from scratch to classify handwritten digits from the MNIST dataset. It avoids external deep learning libraries like TensorFlow or Keras, relying on fundamental numerical computing libraries like NumPy (numpy) for array manipulation and mathematical operations.

## Key Functionalities:

## Data Preprocessing:

- Loads the MNIST dataset using tf.keras.datasets.mnist.load_data().
- Flattens 28x28 image data into 784-dimensional vectors.
- Combines labels and images into DataFrames using Pandas (pandas).
- Shuffles the data to prevent bias during training.
- One-hot encodes labels using a custom one_hot_encode function.
- Normalizes pixel values to the range [0, 1] for better convergence.


## Neural Network Architecture:
- Defines a two-layer network with an input layer, a hidden layer with 10 neurons, and an output layer with 10 neurons (one for each digit class).
- Employs ReLU (Rectified Linear Unit) as the activation function in the hidden layer.
- Uses softmax activation in the output layer to generate class probabilities.

## Learning Algorithm:
- Implements gradient descent for parameter optimization.
- Defines separate functions for forward propagation, backpropagation, and parameter updates.
- Calculates gradients using the chain rule during backpropagation.
- Updates weights and biases based on the learning rate (alpha).

## Training and Evaluation:
- Trains the network for a specified number of iterations.
- Calculates and prints accuracy during training to monitor progress.
- Provides a get_accuracy function to evaluate the model's performance.

## Prediction:
- Defines functions for making predictions on new data (make_predictions) and visualizing test images (test_prediction).
- Allows visualization of a specific image and its predicted class along with the actual label.

## Further Enhancements:
- Experiment with different network architectures (number of layers, neurons).
- Implement more advanced activation functions.
- Explore regularization techniques (e.g., L1/L2 regularization) to prevent overfitting.
- Optimize hyperparameters (learning rate, batch size) for better performance.
- Consider using mini-batch training for improved efficiency.
- Add error handling and logging for better debugging and monitoring.

## Additional Notes:
- This code serves as a foundational example of building a neural network from scratch.
- For more complex tasks or large datasets, consider using established deep learning libraries.
- This implementation can be extended to solve various classification problems with appropriate data preprocessing and output layer configuration.

Thanks.

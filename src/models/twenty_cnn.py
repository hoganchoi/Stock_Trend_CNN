## Creates the design of a CNN model for stock price trend across twenty days. 

## Import necessary packages.
from keras import Sequential, layers

## Initializes the model where twenty days is the input.
def initialize_model(input_shape = (64, 60, 1), num_classes = 1):
    '''
    Creates and returns the design for a twenty day CNN model.

    Args:
        input_shape (tuple): The shape of the input image, defaulted to (64, 60, 1).
        num_classes (int): The number of classes in the output neuron. Defaulted to 1 since model uses 
                            binary classification.

    Returns:
        model (Sequential): Returns the sequential design of the twenty day model.
    '''
    ## Create sequential linear stack for our CNN model.
    model = Sequential()

    ## Rescale the layer to normalize pixel values within range [0, 1].
    model.add(layers.Rescaling(scale = 1./255, input_shape = input_shape))

    ## Create the first 2D Convolutional layer.
    model.add(layers.Conv2D(filters = 64, kernel_size = (5, 3), padding = 'same', strides = (3, 1), dilation_rate = (1, 1),\
                                    input_shape = input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.01))
    ## Create the first Max Pooling layer.
    model.add(layers.MaxPooling2D(pool_size = (2, 1)))

    ## Create the second 2D Convolutional layer.
    model.add(layers.Conv2D(filters = 128, kernel_size = (5, 3), padding = 'same', strides = (1, 1), dilation_rate = (1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.01))
    ## Create the second Max Pooling layer.
    model.add(layers.MaxPooling2D(pool_size = (2, 1)))

    ## Create the third 2D Convolutional layer.
    model.add(layers.Conv2D(filters = 256, kernel_size = (5, 3), padding = 'same', strides = (1, 1), dilation_rate = (1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha = 0.01))
    ## Create the third Max Pooling layer.
    model.add(layers.MaxPooling2D(pool_size = (2, 1)))

    ## Flatten the output from the convolutional layer.
    model.add(layers.Flatten())

    ## Create the fully-connected layer.
    model.add(layers.Dense(units = 46080))
    model.add(layers.LeakyReLU(alpha = 0.01))
    ## Perform dropout regularization on fully-connected layer.
    model.add(layers.Dropout(rate = 0.5))

    ## Create final softmax layer.
    model.add(layers.Dense(units = num_classes, activation = 'softmax'))

    ## Return initialized model.
    return model
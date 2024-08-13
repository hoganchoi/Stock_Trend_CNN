## Creates and returns a certain model given the input days.

## Import necessary packages.
from keras.optimizers import Adam
import src.models.five_cnn as fm
import src.models.twenty_cnn as tm
import src.models.sixty_cnn as sm

## A class representing the entire stock price predictor model.
class StockCNNModel():
    '''
    A class that is able to generate and store any model type (5-days, 20-days, 60-days).

    Attributes:
        type (string): A string representing which type of model the user wishes to train.
        model (Sequential): The Keras generated model.
        optimizer (optimizers): The optimizing function that'll be used for learning.
        loss (string): A string representing the loss function.
        metric (string): A string representing the metric function.

    Methods:
        generate_model(self): Generates and compiles a certain model given the model type.

    Usage:
        Used to generate any of the three types of model used.
    '''
    ## Initializes instance.
    def __init__(self, model_type):
        '''
        Initializes the StockCNNModel class.

        Args:
            model_type (string): A string representing the type of model that the user wants to use.

        Returns:
            None
        '''
        self.type = model_type
        self.model = None
        self.optimizer = Adam(learning_rate = 1e-5)
        self.loss = 'binary_crossentropy'
        self.metric = 'accuracy'

    ## Creates specified model.
    def generate_model(self):
        '''
        Generates, compiles, and returns one of the three models.

        Args:
            None

        Returns:
            self.model (Sequential): The model that the user specified.
        '''
        ## If model type is five days, initialize and compile a five day model.
        if ('five').lower() == (self.type).lower():
            self.model = fm.initialize_model()
            self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = [self.metric])
        
        ## If model type is twenty days, initialize and compile a twenty day model.
        if ('twenty').lower() == (self.type).lower():
            self.model = tm.initialize_model()
            self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = [self.metric])

        ## If model type is sixty days, initialize and compile a sixty day model.
        if ('sixty').lower() == (self.type).lower():
            self.model = sm.initialize_model()
            self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = [self.metric])

        ## Returns compiled model.
        return self.model

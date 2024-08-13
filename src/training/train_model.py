## Compiles model and trains the given model over a specific number of epochs. Returns/plots the accuracy
## of the validation dataset over the epochs and save trained model.

## Import necessary packages.
import tensorflow as tf
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import src.models.cnn_model as cm
import src.utils.file_operations as fo

## Preprocess the images for training and validation.
def preprocess_training_images(train_img_dir):
    '''
    Given the directory for the training images, preprocess and split dataset.
    Args:
        train_img_dir (string): A string representing the training dataset directory.
    Returns:
        train_ds (Dataset): A Tensorflow Dataset for training images.
        val_ds (Dataset): A Tensorflow Dataset for validation images.
        model_type (string): A string representing the model type used for training.
    '''
    ## Initialize the parameters
    model_type = None
    input_shape = None
    interval = train_img_dir.split('\\')[-3]

    ## Determine the model type used for training.
    if int(interval[1]) == 5:
        model_type = 'five'
        input_shape = (32, 15)
    if int(interval[1] == 20):
        model_type = 'twenty'
        input_shape = (64, 60)
    if int(interval[1] == 60):
        model_type = 'sixty'
        input_shape = (96, 180)

    ## Create training and validation dataset using Tensorflow.
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        train_img_dir, 
        labels = 'inferred', 
        label_mode = 'binary', 
        color_mode = 'grayscale', 
        batch_size = 128, 
        image_size = input_shape, 
        shuffle = True, 
        validation_split = 0.3, 
        subset = 'both', 
        seed = 42
    )

    ## Return dataset and model type.
    return train_ds, val_ds, model_type

## Train a certain number of models (5 by default) and save them to designated directory.
def train_models(train_img_dir, output_dir, num_models = 5):
    '''
    Given the input and output directory, trains and saves five separate models.
    Args:
        train_img_dir (string): A string representing the directory storing training images.
        output_dir (string): A string representing the output directory for the models.
        num_models (int): The number of models to train and save (used for ensemble voting).
    Returns:
        None
    '''
    ## Define the number of days and image type.
    interval = train_img_dir.split('\\')[-3]
    img_type = train_img_dir.split('\\')[-2]

    ## Create training and validation dataset along with the model type.
    train_ds, val_ds, model_type = preprocess_training_images(train_img_dir)

    ## Define the early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss', 
        patience = 2, 
        restore_best_weights = True
    )

    ## Train a certain number of models and save them to designated directory.
    for i in range(num_models):
        temp_model = cm.StockCNNModel(model_type).generate_model()
        temp_model.fit(train_ds, epochs = 50, callbacks = [early_stopping], validation_data = val_ds)
        model_dir = os.path.join(output_dir, f'{interval}\\{img_type}\\model_{i}.h5')
        fo.save_model(temp_model, model_dir)

def main():
    train_img_dir = os.path.join('.', 'data', 'processed_img', 'stock_img', 'I5R5', 'base_img', 'training_dataset')
    models_out_dir = os.path.join('.', 'data', 'saved_models')
    train_models(train_img_dir = train_img_dir, output_dir = models_out_dir)

if __name__ == "__main__":
    main()
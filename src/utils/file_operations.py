## Houses file operation functions that are used in other modules.

## Import necessary packages.
import pandas as pd
import os
import matplotlib.pyplot as plt

## Saves a csv file given an output path directory.
def write_csv_file(file, file_path):
    '''
    Saves a csv file given a list.
    
    Args:
        file (List): A list containing elements.
        file_path (string): A string representing the output path directory.
    Returns:
        None
    '''
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    df_file = pd.DataFrame(file)
    df_file.to_csv(file_path, index = False, header = False)

## Saves images given output file path.
def save_img(img, file_path):
    '''
    Given PIL Image, saves it to output directory.
    Args:
        img (Image): A PIL Image.
        file_path (string): A string representing save path.
    Returns:
        None
    '''
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    img.save(file_path)

## Saves the model's weights to given output directory.
def save_model(keras_model, output_path):
    '''
    Saves the trained model to output path.
    Args:
        keras_model (model): A Keras model that has been trained. 
        output_path (string): A string representing the output directory.
    Returns:
        None
    '''
    directory = os.path.dirname(output_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    keras_model.save(output_path)
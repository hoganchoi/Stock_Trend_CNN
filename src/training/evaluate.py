## Import necessary packages.
from collections import defaultdict
import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import src.utils.file_operations as fo

## Creates a dictionary storing images based on ticker name.
def create_ticker_dict(img_dir):
    '''
    Generates a dictionary with each ticker storing its respective images.
    Args:
        img_dir (string): A string representing the image directory.
    Returns:
        ticker_dict (Dictionary): A dictionary with a list of images for each ticker.
    '''
    ## Create a list of image paths from img_dir.
    img_list = os.listdir(img_dir)

    ## Create the dictionary using defaultdict.
    ticker_dict = defaultdict(list)

    ## For each image name in the list, append to the dictionary.
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        ticker = img_name.split('_')[0]
        ticker_dict[ticker].append(img_path)

    ## Sort the list in numerical order.
    for name in ticker_dict:
        ticker_dict[name].sort(key = lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    ## Return the dictionary.
    return ticker_dict    

## Generates an image array given image path.
def get_img_arr(img_path):
    '''
    Given image path, creates its respective image array.
    Args:
        img_path (string): The path to OHLC image.
    Returns:
        img_arr (np.array): An image array created using Tensorflow.
    '''
    ## Use Tensorflow's preprocessing function to create image array from image path.
    img = tf.keras.preprocessing.image.load_img(img_path, color_mode = 'grayscale')
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    return img_arr

## Create an image dictionary.
def create_img_dict(img_dir):
    '''
    Create a dictionary storing all the images given the image directory.
    Args:
        img_dir (string): A string representing the path to all images
    Returns:
        img_dict (Dictionary): A dictionary storing all images to its respective ticker.
    '''
    ## Create a ticker dictionary.
    ticker_dict = create_ticker_dict(img_dir)

    ## Createa default dictionary and a counter.
    img_dict = defaultdict(list)
    counter = 1

    ## For each ticker's image in the ticker dictionary, create an image array and 
    ## append to new image dictionary.
    for key, value in ticker_dict.items():
        for img_path in value:
            img_arr = get_img_arr(img_path)
            img_dict[key].append(img_arr)
        print(f"Finished {counter} / {len(ticker_dict)}")
        counter = counter + 1

    ## Return created image dictionary.
    return img_dict

## Calculate the minimum number of images in image dictionary.
def get_min_length(img_dict):
    '''
    Finds the minimum number of weeks within image dictionary.
    Args:
        img_dict (Dictionary): A dictionary containing all OHLC images for each firm.
    Returns:
        min_len (int): The minimum length of weeks for listed firms.
    '''
    ## Initialize the minimum length
    min_len = np.inf

    ## Search through all sections of image dictionary to find minimum length of weeks.
    for key, value in img_dict.items():
        ## If the number of images is less than the minimum length, assign to variable.
        if len(value) < min_len:
            min_len = len(value)

    ## Return the minimum length of images in the image dictionary.
    return min_len

## Use a model to predict given images.
def get_prediction(img_dict, model_dir):
    '''
    Predict images given a model.
    Args:
        img_dict (Dictionary): A dictionary containing all OHLC images for each listed firm.
        model_dir (string): A string representing the directory for the model.
    Returns:
        predictions (List): A list containing all the predictions for individual firms.
    '''
    ## Load in model weights from model directory.
    model = tf.keras.models.load_model(model_dir)

    ## Intiliaze prediction list and a count.
    predictions = []
    counter = 1

    ## Get the minimum length of images from image dictionary.
    min_len = get_min_length(img_dict)

    ## For each firm listed in dictionary, apply the model to the images.
    for key, value in img_dict.items():
        ## Clear session every iteration to save memory.
        tf.keras.backend.clear_session()

        ## Use the model to predict all the images listed for firm.
        prediction = model.predict(np.array(value), batch_size = 16)

        ## If the prediction shape is greater than minimum length, remove first week.
        if prediction.shape[0] > min_len:
            prediction = prediction[1:]

        ## Append the model's predictions to list.
        predictions.append(prediction)
        print(f'{counter} / {len(img_dict)} Finished')
        counter = counter + 1

    ## Delete the model to save memory.
    del model

    ## Return the list of predictions for each firm.
    return predictions

## Perform ensemble prediction given the directory to models.
def get_avg_prediction(img_dict, models_dir):
    '''
    Applies all the models in given directory to obtain an average prediction.
    Args:
        img_dict (Dictionary): A dictionary containing all the OHLC images for individual firms.
        models_dir (string): A string representing the directory for all models.
    Returns:
        pred_avg (np.array): 
    '''
    ## Obtain the model list from directory.
    models = os.listdir(models_dir)

    ## Initializes prediction list and counter.
    pred_list = []
    counter = 1

    ## For each model in model list, compute the predicted probability for each image.
    for model_name in models:
        ## Get the path to specific model.
        model_path = os.path.join(models_dir, model_name)

        ## Get the predicted probabilities for specific model.
        temp_preds = get_prediction(img_dict, model_path)

        ## Append the predicted values to prediction list.
        pred_list.append(temp_preds)
        print(f"Finished {counter} / {len(models)}")
        counter = counter + 1

    ## Conver the list to array and find average across the five models.
    pred_arr = np.array(pred_list)
    pred_avg = np.mean(pred_arr, axis = 0)

    ## Return the averaged predicted probabilities.
    return pred_avg

## Save the calculated average predictions to given directory.
def save_predictions(img_dict, pred_avg, output_dir):
    '''
    Given output directory, save the average predictions as a csv file for each firm.
    Args:
        img_dict (Dictionary): A dictionary containing OHLC images for all firms.
        pred_avg (np.array): An array containing the average probabilities for each image.
        output_dir (string): A string representing the output directory for the predicted values.
    Returns:
        None
    '''
    ## Obtain a list of all tickers from the image dictionary.
    ticker_list = list(img_dict.keys())

    ## Initialize count.
    counter = 1

    ## For each firm in the ticker list, extract their probability and save it as a csv file.
    for i in range(len(ticker_list)):
        ## Obtain the prediction csv file name.
        pred_name = f"{ticker_list[i]}_pred.csv"

        ## Create the file path directory with created name.
        pred_dir = os.path.join(output_dir, pred_name)

        ## Get the predicted average and write a csv file to designated path.
        pred_data = pred_avg[i].tolist()
        fo.write_csv_file(pred_data, pred_dir)
        print(f"{counter} / {len(ticker_list)} Finished")
        counter = counter + 1

## Creates an arrary of returns and their respective probability.
def returns_based_predictions(ret_dir, pred_dir):
    '''
    Returns the weekly returns and their respective probability predicted by the model.
    Args:
        ret_dir (string): The string representing the directory storing all the returns.
        pred_dir (string): The string representing the directory storing all the probabilities.
    Returns:
        ret_stack (np.array): An array of weekly returns, where weeks are rows and firm is columns.
        pred_stack (np.array): An array of predicted probabilities, where weeks are rows and firm is columns.
    '''
    ## Obtain respective list for weekly returns and probabilites.
    ret_list = os.listdir(ret_dir)
    pred_list = os.listdir(pred_dir)

    ## Initialize prediction array size (must match the return array).
    pred_len = 0

    ## Initialize empty arrays for returns and predictions.
    ret_stack = []
    pred_stack = []

    ## Load data from csv files (returns and predictions) and horizontally stack together.
    for i in range(len(ret_list)):
        ## Get specific return csv path.
        ret_name = ret_list[i]
        ret_path = os.path.join(ret_dir, ret_name)

        ## Get specific prediction csv path.
        pred_name = pred_list[i]
        pred_path = os.path.join(pred_dir, pred_name)

        ## Load in csv files for both returns and predictions.
        ret_arr = np.loadtxt(ret_path, delimiter = ',').reshape(-1, 1)
        pred_arr = np.loadtxt(pred_path, delimiter = ',').reshape(-1, 1)

        ## Define length for prediction.
        pred_len = pred_arr.shape[0]

        ## Make sure to remove first week if length is greater than prediction length.
        if ret_arr.shape[0] > pred_len:
            ret_arr = ret_arr[1:]

        ## Append returns and predictions array to list.
        ret_stack.append(ret_arr)
        pred_stack.append(pred_arr)

    ## Horizontally stack the arrays together so that weeks are rows and firms are columns.
    ret_stack = np.hstack(ret_stack)
    pred_stack = np.hstack(pred_stack)

    ## Return both stacked arrays.
    return ret_stack, pred_stack

## Sort the weekly returns based on their predicted probability.
def sort_returns(ret_stack, pred_stack):
    '''
    Sort the weekly returns based on their probability from the model.
    Args:
        ret_stack (np.array): A stacked array containing weekly returns.
        pred_stack (np.array): A stacked array containing the predicted probabilities.
    Returns:
        ret_sorted (np.array): A sorted stacked array containing weekly returns.
    '''
    ## Create a copy of weekly returns.
    ret_sorted = ret_stack.copy()

    ## For each week, sort each firm's returns based on probability.
    for i in range(len(ret_stack)):
        ## Get respective weekly row from both arrays.
        ret_row = ret_stack[i]
        pred_row = pred_stack[i]

        ## Get the indices of sorted probabilities (low to high). 
        pred_row_sorted = np.argsort(pred_row)

        ## Use sorted indices on returns and save to the new sorted array.
        ret_row_sorted = ret_row[pred_row_sorted]
        ret_sorted[i] = ret_row_sorted

    ## Return sorted array.
    return ret_sorted

## Get the average annual returns from returns.
def get_comp_return(ret_sorted, num_years = 7):
    '''
    Calculate the Compound Annual Growth Rate for overall returns.
    Args:
        ret_sorted (np.array): The sorted array of weekly returns.
        num_years (int): The number of years for evaluating model.
    Returns:
        ret_df (Dataframe): A dataframe containing annual growth rate.
    '''
    ## Add 1 to all returns to get overall growth for each predicted firm.
    growth = 1 + ret_sorted

    ## Calculate the product of growths over a period of time.
    tot_growth = np.prod(growth, axis = 0)

    ## Obtain the annual growth rate by applying the CAGR formula.
    ann_ret = (np.sign(tot_growth) * (np.abs(tot_growth) ** (1 / num_years))) - 1

    ## Save the overall annual returns as a Dataframe.
    ret_df = pd.DataFrame(ann_ret, columns = ['Returns'])

    ## Return the annual returns Dataframe
    return ret_df
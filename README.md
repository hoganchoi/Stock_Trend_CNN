# Stock Neural Network Project

## Summary
In this project, I aimed to re-create the Convolutional Neural Network (CNN) models proposed in the paper "(Re-)Imag(in)ing Price Trends" by Jiang, Kelly, and Xiu. This paper approaches trend-based predictability in stock prices by training CNN models to learn the price patterns in OHLC graphs. This project showcases the following:
 - Data acquisition from public sources
 - Data preprocessing
 - Generating custom made OHLC images
 - Creating and compiling Neural Network models using Keras
 - Making decile portfolios to evaluate the performance of created models

Due to the model's complex architecture, I was only able to compile and train the five day CNN models. While the twenty day and sixty day CNN models' design are included in this project, I wasn't able to train these models. Furthermore, the training process is very computationally expensive. Hence, as of right now, I've only included the results for the five day CNN model on base images. The performance of the model on other variations of OHLC images will later be added.

Also, please note that this project was developed to demonstrate the models from the paper and may be lacking in practicality.

## Requirements
This project was created in `Python=3.10`. The additional requirements needed to compile this code correctly are listed in the `requirements.txt` located in the root folder. 

## USEME Location
The `USEME.md` gives detailed explanation on how to run the scripts provided in this project. The `USEME.md` file is located in the root folder.

## MORE_INFO Location
The `MORE_INFO.md` file gives detailed explanation into the images used for this project, the model architecture, and the performance results of the model. The `MORE_INFO.md` is located in the root folder. 

## Dataset Availability
This project comes with a `csv` file containing the list of firms that I worked with, OHLC images, and pre-trained model weights (I5R5 base image only). However, due to the large number of images needed for this project (a single dataset can potentially contain approximately 1.5 million images), I've provided a zip file in the data directory storing some of the datasets. I've uploaded the list of firms and the pre-trained model weights; the rest of the datasets are contained the provided zip file. Hence, if you want to run this project, please use the scripts to generate the images from the provided `csv` file or extract the zip file before running the notebooks.

## Example Images
There are six images in the 'example_images' directory. This directory stores the images used in the `MORE_INFO.md` file.

## Code Design
This project consists of the `src` folder, `data` folder, three Python notebooks, and a list of firms used for this project. 

The `src` folder contains the source code for the project. It has the following packages:
 - `data`: Stores all the code for data acquisition and image generation.
    - `get_data`: Module containing code for obtaining and preprocessing data.
    - `get_images`: Module containing code for generating custom made OHLC images.
 - `models`: Stores code for initializing and compiling CNN models.
    - `cnn_model`: A general module that allows users to choose which model to deploy and compile.
    - `five_cnn`: Code for initializing the CNN model for five day OHLC images.
    - `twenty_cnn`: Code for initializing the CNN model for twenty day OHLC images.
    - `sixty_cnn`: Code for initializing the CNN model for sixty day OHLC images.
    
    (NOTE: Five, twenty, and sixty business days refer to one week, one month, and three months of stock period)
 - `training`: Stores all the code necessary for training the model and evaluating its performance.
    - `train_model`: Contains the code for training and saving the models.
    - `evaluate`: Contains the code for evaluating the performance of the trained models.
 - `utils`: Stores code for utility functions and helper modules for the project.
    - `file_operations`: Module containing all the code related to file operations (saving, loading, etc).

The `data` folder stores all the datasets and images compiled by the project.
 - `raw_dataset`: Contains the preprocessed stock data from compiling `get_data`.
    - `artificial_data`: Contains stock information from artificially created stock trends.
    - `stock_data`: Contains time-series stock information from a list of firms.
 - `processed_img`: Contains all the custom made OHLC images that can be used for training and testing the models.
    - `artificial_img`: Stores all the OHLC images for the artifically created stock trends.
    - `stock_img`: Stores all the OHLC images from the stock trends of the provided list of firms.
        - `I5R5`: Contains the OHLC images for one week stock trends and predictions for the next week.
            - `base_img`: The original base OHLC graph images.
               - `testing_dataset`: Contains all the testing images.
               - `training_dataset`: Contains all the training images.
                  - `label_0`: All OHLC images that decrease in stock price the following week.
                  - `label_1`: All OHLC images that increase in stock price the following week.
            - `ma_img`: The OHLC graph images with a moving average line.
            - `vol_img`: The OHLC graph images with the volume bars.
            - `vol_ma_img`: The OHLC graph images with the moving average line and volume bars.
            - `returns`: Contains csv files for the weekly returns for each firm.
        - `I5R20`: Contains the OHLC images for one week stock trends and predictions for the next month.
        - `I5R60`: Contains the OHLC images for one week stock trends and predictions for the next three months. 
        - `I20R5`: Contains the OHLC images for one month stock trends and predictions for the next week.
        - `I20R20`: Contains the OHLC images for one month stock trends and predictions for the next month. 
        - `I20R60`: Contains the OHLC images for one month stock trends and predictions for the next three months.
        - `I60R5`: Contains the OHLC images for three months stock trends and predictions for the next one week.
        - `I60R20`: Contains the OHLC images for three months stock trends and predictions for the next month.
        - `I60R60`: Contains the OHLC images for three months stock trends and predictions for the next three months.
 - `saved_models`: Contains trained model weights for one week, one month, and three months stock trend images.
 - `predictions`: Contains the predictions of train models for each individual firm (one week, one month, and three months stock trend).

The `Create_Images.ipynb` is a notebook documenting the process of acquiring stock trend datasets, preprocessing the obtained datasets, and generating OHLC images.

The `Train_Model.ipynb` is a notebook documenting the process of training and saving the CNN models.

The `Model_Evaluation.ipynb` is a notebook documenting the process of loading in trained models, applying models to testing dataset to make predictions, and evaluating the overall performance of the models.

The `firm_list.csv` file stores all the tickers that were used to create the OHLC images. 

## References

## Creates trainable images and respective labels from the generated raw datasets (csv files). 
## These images are saved in the 'processed_img' folder. 

## Import necessary packages.
import pandas as pd
import math
from PIL import Image, ImageDraw
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import src.utils.file_operations as fo

## Create dataset and labels for input and return days.
def make_interval_returns(stock_data, num_days, return_days):
    '''
    Takes in the stock data, the input window frame, and the return window frame
    to create the initial training data and true labels for data.
    Args:
        stock_data (Dataframe): The dataframe for stock prices.
        num_days (int): The input days that are used for training.
        return_days (int): The number of days later that the model predicts.
    Returns:
        day_data (List): A list containing the dataframe for input days (5, 20, 60).
        data_labels (List): A list containing whether the stock price will go up
                            or not in a specific number of days.
        date_list (List): A list containing the year of each interval.
        return_list (List): A list containing all the return values for each interval.
    '''
    day_data = []
    data_labels = []
    date_list = []
    return_list = []

    for i in range(0, len(stock_data), num_days):
        date_list.append(stock_data['Date'].dt.year.iloc[i])

        input_close_index = i + num_days - 1
        return_close_index = input_close_index + return_days

        if return_close_index < len(stock_data):
            input_close = stock_data['Close Norm'].iloc[input_close_index]
            return_close = stock_data['Close Norm'].iloc[return_close_index]

            tot_ret = (return_close - input_close) / input_close
            return_list.append(tot_ret)

            day_data.append(stock_data[i:i + num_days])

            if (return_close > input_close):
                data_labels.append(1)

            else:
                data_labels.append(0)

        else:
            break

    return day_data, data_labels, date_list, return_list


## Draw the OHLC graph given stock data for a certain time frame window.
def draw_ohlc_graph(stock_data, num_days, is_volume = False, is_moving_avg = False):
    '''
    Given a stock data for a certain time frame window (5, 20, 60 days), draw an
    OHLC chart. These images will be used for training.
    Args:
        stock_data (Dataframe): Dataframe for stock prices for certain time frame
                                window.
        num_days (int): The number of days for input time frame window (5, 20, 60).
        is_volume (bool): Whether to include volume in the OHLC chart.
        is_moving_avg (bool): Whether to include moving average in the OHLC chart.
    Returns:
        (Image): An image of an OHLC chart for a certain time frame window.
    '''
    ## One day is 3 pixels in width.
    img_width = 3 * num_days

    ## These are the heights in pixels for each respective day (noted in paper).
    if num_days == 5:
        img_height = 32
    if num_days == 20:
        img_height = 64
    if num_days == 60:
        img_height = 96

    ## Create a grayscale image using Pillow.
    graph_img = Image.new('1', (img_width, img_height), 'black')
    graph_draw = ImageDraw.Draw(graph_img)

    ## If volume is included, adjust the stock price height and the volume height
    ## (noted in paper).
    if is_volume:
        stock_height = math.ceil((img_height - 1) * 0.8)
        vol_height = math.floor((img_height - 1) * 0.2)

        max_vol = stock_data['Volume'].max()
    else:
        stock_height = img_height

    ## If moving average is included, adjust maximum and minimum price for proper
    ## stock range.
    if is_moving_avg:
        min_price = stock_data[['Open Norm', 'High Norm', 'Close Norm', 'Low Norm', 'Moving Avg']].min().min()
        max_price = stock_data[['Open Norm', 'High Norm', 'Close Norm', 'Low Norm', 'Moving Avg']].max().max()
    else:
        min_price = stock_data[['Open Norm', 'High Norm', 'Close Norm', 'Low Norm']].min().min()
        max_price = stock_data[['Open Norm', 'High Norm', 'Close Norm', 'Low Norm']].max().max()

    stock_range = max_price - min_price

    ## If there's no change in price, draw a horizontal line through the graph.
    if stock_range == 0:
        for i in range(len(stock_data)):
            ## Create x-coordinate for current day.
            x = i * 3

            ## If volume is included, draw volume bar and add 1 pixel margin between
            ## chart and volume bar.
            if is_volume:
                ## If there was no volume traded that day, mark it 0.
                if max_vol == 0:
                    volume_point = 0

                    vol_space = 1 + vol_height

                    graph_draw.line([(x + 1, 0), (x + 1, volume_point)], fill = 1)
                else:
                    vol_space = 1 + vol_height

                    volume = stock_data['Volume'].iloc[i]
                    volume_point = int((volume / max_vol) * (vol_height - 1))

                    graph_draw.line([(x + 1, 0), (x + 1, volume_point)], fill = 1)
            else:
                vol_space = 0

            ## Create y-coordinate for each Open, High, Low, and Close prices.
            open_point = int(img_height / 2) + vol_space
            high_point = int(img_height / 2) + vol_space
            low_point = int(img_height / 2) + vol_space
            close_point = int(img_height / 2) + vol_space

            ## Draw each point for Open, High, Low, and Close prices in chart.
            graph_draw.line([(x, open_point), (x + 1, open_point)], fill = 1)
            graph_draw.line([(x + 1, high_point), (x + 1, low_point)], fill = 1)
            graph_draw.line([(x + 1, close_point), (x + 2, close_point)], fill = 1)

            ## If moving average is included, create and draw the line in the chart.
            if is_moving_avg and (i + 1) < len(stock_data):
                moving_avg_price1 = stock_data['Moving Avg'].iloc[i]
                moving_avg_price2 = stock_data['Moving Avg'].iloc[i + 1]

                moving_avg_point1 = int(img_height / 2) + vol_space
                moving_avg_point2 = int(img_height / 2) + vol_space

                graph_draw.line([(x + 1, moving_avg_point1), (x + 4, moving_avg_point2)], fill = 1)

    else:
        ## For each day, draw an OHLC point.
        for i in range(len(stock_data)):
            ## Obtain prices at each Open, High, Low, and Close points.
            open_price = stock_data['Open Norm'].iloc[i]
            high_price = stock_data['High Norm'].iloc[i]
            low_price = stock_data['Low Norm'].iloc[i]
            close_price = stock_data['Close Norm'].iloc[i]

            ## Create x-coordinate for current day.
            x = i * 3

            ## If volume is included, draw volume bar and add 1 pixel margin between
            ## chart and volume bar.
            if is_volume:
                if max_vol == 0:
                    vol_space = 1 + vol_height

                    volume_point = 0

                    graph_draw.line([(x + 1, 0), (x + 1, volume_point)], fill = 1)
                else:
                    vol_space = 1 + vol_height

                    volume = stock_data['Volume'].iloc[i]
                    volume_point = int((volume / max_vol) * (vol_height - 1))

                    graph_draw.line([(x + 1, 0), (x + 1, volume_point)], fill = 1)
            else:
                vol_space = 0

            ## Create y-coordinate for each Open, High, Low, and Close prices.
            open_point = int(((open_price - min_price) / stock_range) * (stock_height - 1)) + vol_space
            high_point = int(((high_price - min_price) / stock_range) * (stock_height - 1)) + vol_space
            low_point = int(((low_price - min_price) / stock_range) * (stock_height - 1)) + vol_space
            close_point = int(((close_price - min_price) / stock_range) * (stock_height - 1)) + vol_space

            ## Draw each point for Open, High, Low, and Close prices in chart.
            graph_draw.line([(x, open_point), (x + 1, open_point)], fill = 1)
            graph_draw.line([(x + 1, high_point), (x + 1, low_point)], fill = 1)
            graph_draw.line([(x + 1, close_point), (x + 2, close_point)], fill = 1)

            ## If moving average is included, create and draw the line in the chart.
            if is_moving_avg and (i + 1) < len(stock_data):
                moving_avg_price1 = stock_data['Moving Avg'].iloc[i]
                moving_avg_price2 = stock_data['Moving Avg'].iloc[i + 1]

                moving_avg_point1 = int(((moving_avg_price1 - min_price) / stock_range) * (stock_height - 1)) + vol_space
                moving_avg_point2 = int(((moving_avg_price2 - min_price) / stock_range) * (stock_height - 1)) + vol_space

                graph_draw.line([(x + 1, moving_avg_point1), (x + 4, moving_avg_point2)], fill = 1)

    ## Transpose entire image before returning the function.
    return graph_img.transpose(Image.FLIP_TOP_BOTTOM)

## Creates training data and their respective labels.
def create_img_labels(stock_data):
    '''
    Creates images and labels for given stock dataset.
    Args:
        stock_data (Dataframe): A dataframe containing stock data.
    Returns:
        training_dict (Dictionary): A dictionary containing stock data for each interval.
        labels_dict (Dictionary): A dictionary containing labels for each respective interval.
        years_dict (Dictionary): A dictionary containing years for each respective interval.
        ret_dict (Dictionary): A dictionary containing returns for each respective interavl.
    '''
    ## Generate stock data for 5 days, their respective predictions and return values for next 5 days, and their years.
    I5R5_data, I5R5_labels, I5R5_years, I5R5_returns = make_interval_returns(stock_data, 5, 5)
    ## Generate stock data for 5 days, their respective predicitions and return values for next 20 days, and their years.
    I5R20_data, I5R20_labels, I5R20_years, I5R20_returns = make_interval_returns(stock_data, 5, 20)
    ## Generate stock data for 5 days, their respective predictions and return values for next 60 days, and their years.
    I5R60_data, I5R60_labels, I5R60_years, I5R60_returns = make_interval_returns(stock_data, 5, 60)

    I5_data_list = [I5R5_data, I5R20_data, I5R60_data]
    I5_label_list = [I5R5_labels, I5R20_labels, I5R60_labels]
    I5_year_list = [I5R5_years, I5R20_years, I5R60_years]
    I5_return_list = [I5R5_returns, I5R20_returns, I5R60_returns]

    ## Generate stock data for 20 days, their respective predictions and return values for next 5 days, and their years.
    I20R5_data, I20R5_labels, I20R5_years, I20R5_returns = make_interval_returns(stock_data, 20, 5)
    ## Generate stock data for 20 days, their respective predictions and return values for next 20 days, and their years.
    I20R20_data, I20R20_labels, I20R20_years, I20R20_returns = make_interval_returns(stock_data, 20, 20)
    ## Generate stock data for 20 days, their respective predictions and return values for next 60 days, and their years.
    I20R60_data, I20R60_labels, I20R60_years, I20R60_returns = make_interval_returns(stock_data, 20, 60)

    I20_data_list = [I20R5_data, I20R20_data, I20R60_data]
    I20_label_list = [I20R5_labels, I20R20_labels, I20R60_labels]
    I20_year_list = [I20R5_years, I20R20_years, I20R60_years]
    I20_return_list = [I20R5_returns, I20R20_returns, I20R60_returns]

    ## Generate stock data for 60 days, their respective predictions and return values for next 5 days, and their years.
    I60R5_data, I60R5_labels, I60R5_years, I60R5_returns = make_interval_returns(stock_data, 60, 5)
    ## Generate stock data for 60 days, their respective predictions and return values for next 20 days, and their years.
    I60R20_data, I60R20_labels, I60R20_years, I60R20_returns = make_interval_returns(stock_data, 60, 20)
    ## Generate stock data for 60 days, their respective predictions and return values for next 60 days, and their years.
    I60R60_data, I60R60_labels, I60R60_years, I60R60_returns = make_interval_returns(stock_data, 60, 60)

    I60_data_list = [I60R5_data, I60R20_data, I60R60_data]
    I60_label_list = [I60R5_labels, I60R20_labels, I60R60_labels]
    I60_year_list = [I60R5_years, I60R20_years, I60R60_years]
    I60_return_list = [I60R5_returns, I60R20_returns, I60R60_returns]

    training_dict = dict(I5_data = I5_data_list, I20_data = I20_data_list, I60_data = I60_data_list)
    labels_dict = dict(I5_labels = I5_label_list, I20_labels = I20_label_list, I60_labels = I60_label_list)
    years_dict = dict(I5_years = I5_year_list, I20_years = I20_year_list, I60_years = I60_year_list)
    ret_dict = dict(I5_returns = I5_return_list, I20_returns = I20_return_list, I60_returns = I60_return_list)

    return training_dict, labels_dict, years_dict, ret_dict

## Creates training images for each interval given volume and moving average.
def create_training_img(data_dict, is_volume, is_moving_avg):
    '''
    From the dictionary containing stock price data, creates individual OHLC images for 
    each interval. Stores images in dictionary.
    Args:
        data_dict (Dictionary): A dictionary containing stock prices for each interval.
        is_volume (bool): Whether to show volume or not.
        is_moving_avg (bool): Whether to show moving average line or not.
    Returns:
        img_dict (Dictionary): A dictionary containing OHLC images for each interval.
    '''
    I5_img_list = []
    I20_img_list = []
    I60_img_list = []

    for key, value in data_dict.items():
        if "5" in key:
            for data in value:
                temp_img_list = []
                for interval in data:
                    temp_img = draw_ohlc_graph(interval, 5, is_volume, is_moving_avg)
                    if temp_img is not None:
                        temp_img_list.append(temp_img)
                I5_img_list.append(temp_img_list)
        if "20" in key:
            for data in value:
                temp_img_list = []
                for interval in data:
                    temp_img = draw_ohlc_graph(interval, 20, is_volume, is_moving_avg)
                    if temp_img is not None:
                        temp_img_list.append(temp_img)
                I20_img_list.append(temp_img_list)
        if "60" in key:
            for data in value:
                temp_img_list = []
                for interval in data:
                    temp_img = draw_ohlc_graph(interval, 60, is_volume, is_moving_avg)
                    if temp_img is not None:
                        temp_img_list.append(temp_img)
                I60_img_list.append(temp_img_list)

    img_dict = dict(I5_img = I5_img_list, I20_img = I20_img_list, I60_img = I60_img_list)
    return img_dict

## Saves OHLC images for given stock.
def save_img(img_dict, label_dict, years_dict, ticker, graph_type, out_dir):
    '''
    Saves each OHLC image to a classification folder for a given stock.
    Args:
        img_dict (Dictionary): The dictionary containing OHLC images for given stock.
        label_dict (Dictionary): The dictionary containing all labels for each interval.
        years_dict (Dictionary): A dictionary containing all years for each interval.
        ticker (string): The ticker representing the firm.
        graph_type (string): The string representing whether image is base, volume, etc.
        out_dir (string): The directory for the output images.
    Returns:
        None
    '''
    for key, value in img_dict.items():
        if "5" in key:
            for i in range(len(value)):
                if i == 0:
                    break_num = 0
                    temp_out_dir = os.path.join(out_dir, f'I5R5\\{graph_type}')
                    for j in range(len(value[i])):
                        label = label_dict['I5_labels'][i][j]
                        if years_dict['I5_years'][i][j] < 2007:
                            img_out_dir = os.path.join(temp_out_dir, f'training_dataset\\label_{label}\\{ticker}_{j}.png')
                        if years_dict['I5_years'][i][j] >= 2007:
                            if break_num == 0:
                                break_num = break_num + j
                            img_out_dir = os.path.join(temp_out_dir, f'testing_dataset\\{ticker}_{j - break_num}.png')
                        fo.save_img(value[i][j], img_out_dir)
                if i == 1:
                    break_num = 0
                    temp_out_dir = os.path.join(out_dir, f'I5R20\\{graph_type}')
                    for j in range(len(value[i])):
                        label = label_dict['I5_labels'][i][j]
                        if years_dict['I5_years'][i][j] < 2007:
                            img_out_dir = os.path.join(temp_out_dir, f'training_dataset\\label_{label}\\{ticker}_{j}.png')
                        if years_dict['I5_years'][i][j] >= 2007:
                            if break_num == 0:
                                break_num = break_num + j
                            img_out_dir = os.path.join(temp_out_dir, f'testing_dataset\\{ticker}_{j - break_num}.png')
                        fo.save_img(value[i][j], img_out_dir)
                if i == 2:
                    temp_out_dir = os.path.join(out_dir, f'I5R60\\{graph_type}')
                    for j in range(len(value[i])):
                        label = label_dict['I5_labels'][i][j]
                        if years_dict['I5_years'][i][j] < 2007:
                            img_out_dir = os.path.join(temp_out_dir, f'training_dataset\\label_{label}\\{ticker}_{j}.png')
                        if years_dict['I5_years'][i][j] >= 2007:
                            if break_num == 0:
                                break_num = break_num + j
                            img_out_dir = os.path.join(temp_out_dir, f'testing_dataset\\{ticker}_{j - break_num}.png')
                        fo.save_img(value[i][j], img_out_dir)
        if "20" in key:
            for i in range(len(value)):
                if i == 0:
                    break_num = 0
                    temp_out_dir = os.path.join(out_dir, f'I20R5\\{graph_type}')
                    for j in range(len(value[i])):
                        label = label_dict['I20_labels'][i][j]
                        if years_dict['I20_years'][i][j] < 2007:
                            img_out_dir = os.path.join(temp_out_dir, f'training_dataset\\label_{label}\\{ticker}_{j}.png')
                        if years_dict['I20_years'][i][j] >= 2007:
                            if break_num == 0:
                                break_num = break_num + j
                            img_out_dir = os.path.join(temp_out_dir, f'testing_dataset\\{ticker}_{j - break_num}.png')
                        fo.save_img(value[i][j], img_out_dir)
                if i == 1:
                    break_num = 0
                    temp_out_dir = os.path.join(out_dir, f'I20R20\\{graph_type}')
                    for j in range(len(value[i])):
                        label = label_dict['I20_labels'][i][j]
                        if years_dict['I20_years'][i][j] < 2007:
                            img_out_dir = os.path.join(temp_out_dir, f'training_dataset\\label_{label}\\{ticker}_{j}.png')
                        if years_dict['I20_years'][i][j] >= 2007:
                            if break_num == 0:
                                break_num = break_num + j
                            img_out_dir = os.path.join(temp_out_dir, f'testing_dataset\\{ticker}_{j - break_num}.png')
                        fo.save_img(value[i][j], img_out_dir)
                if i == 2:
                    break_num = 0
                    temp_out_dir = os.path.join(out_dir, f'I20R60\\{graph_type}')
                    for j in range(len(value[i])):
                        label = label_dict['I20_labels'][i][j]
                        if years_dict['I20_years'][i][j] < 2007:
                            img_out_dir = os.path.join(temp_out_dir, f'training_dataset\\label_{label}\\{ticker}_{j}.png')
                        if years_dict['I20_years'][i][j] >= 2007:
                            if break_num == 0:
                                break_num = break_num + j
                            img_out_dir = os.path.join(temp_out_dir, f'testing_dataset\\{ticker}_{j - break_num}.png')
                        fo.save_img(value[i][j], img_out_dir)
        if "60" in key:
            for i in range(len(value)):
                if i == 0:
                    break_num = 0
                    temp_out_dir = os.path.join(out_dir, f'I60R5\\{graph_type}')
                    for j in range(len(value[i])):
                        label = label_dict['I60_labels'][i][j]
                        if years_dict['I60_years'][i][j] < 2007:
                            img_out_dir = os.path.join(temp_out_dir, f'training_dataset\\label_{label}\\{ticker}_{j}.png')
                        if years_dict['I60_years'][i][j] >= 2007:
                            if break_num == 0:
                                break_num = break_num + j
                            img_out_dir = os.path.join(temp_out_dir, f'testing_dataset\\{ticker}_{j - break_num}.png')
                        fo.save_img(value[i][j], img_out_dir)
                if i == 1:
                    break_num = 0
                    temp_out_dir = os.path.join(out_dir, f'I60R20\\{graph_type}')
                    for j in range(len(value[i])):
                        label = label_dict['I60_labels'][i][j]
                        if years_dict['I60_years'][i][j] < 2007:
                            img_out_dir = os.path.join(temp_out_dir, f'training_dataset\\label_{label}\\{ticker}_{j}.png')
                        if years_dict['I60_years'][i][j] >= 2007:
                            if break_num == 0:
                                break_num = break_num + j
                            img_out_dir = os.path.join(temp_out_dir, f'testing_dataset\\img{ticker}__{j - break_num}.png')
                        fo.save_img(value[i][j], img_out_dir)
                if i == 2:
                    break_num = 0
                    temp_out_dir = os.path.join(out_dir, f'I60R60\\{graph_type}')
                    for j in range(len(value[i])):
                        label = label_dict['I60_labels'][i][j]
                        if years_dict['I60_years'][i][j] < 2007:
                            img_out_dir = os.path.join(temp_out_dir, f'training_dataset\\label_{label}\\{ticker}_{j}.png')
                        if years_dict['I60_years'][i][j] >= 2007:
                            if break_num == 0:
                                break_num = break_num + j
                            img_out_dir = os.path.join(temp_out_dir, f'testing_dataset\\{ticker}_{j - break_num}.png')
                        fo.save_img(value[i][j], img_out_dir)
    print(f"{ticker} {graph_type} Images Saved Successfully!")

## Saves the return values for each interval.
def save_returns(ret_dict, years_dict, ticker, out_dir):
    '''
    Given the return and years dictionary, save the return values for all testing datasets.
    Args:
        ret_dict (Dictionary): The dictionary storing all the return values for each interval.
        years_dict (Dictionary): The dictionary storing all the years for each interval.
        ticker (string): The string representing the ticker of the firm.
        out_dir (string): The string representing the output directory.
    Returns:
        None
    '''
    for key, value in ret_dict.items():
        if "5" in key:
            for i in range(len(value)):
                testing_ret = []
                if i == 0:
                    ret_out_dir = os.path.join(out_dir, f'I5R5\\returns\\{ticker}_returns.csv')
                    for j in range(len(value[i])):
                        if years_dict['I5_years'][i][j] >= 2007:
                            testing_ret.append(ret_dict['I5_returns'][i][j])
                    fo.write_csv_file(testing_ret, ret_out_dir)
                if i == 1:
                    ret_out_dir = os.path.join(out_dir, f'I5R20\\returns\\{ticker}_returns.csv')
                    for j in range(len(value[i])):
                        if years_dict['I5_years'][i][j] >= 2007:
                            testing_ret.append(ret_dict['I5_returns'][i][j])
                    fo.write_csv_file(testing_ret, ret_out_dir)
                if i == 2:
                    ret_out_dir = os.path.join(out_dir, f'I5R60\\returns\\{ticker}_returns.csv')
                    for j in range(len(value[i])):
                        if years_dict['I5_years'][i][j] >= 2007:
                            testing_ret.append(ret_dict['I5_returns'][i][j])
                    fo.write_csv_file(testing_ret, ret_out_dir)
        if "20" in key:
            for i in range(len(value)):
                testing_ret = []
                if i == 0:
                    ret_out_dir = os.path.join(out_dir, f'I20R5\\returns\\{ticker}_returns.csv')
                    for j in range(len(value[i])):
                        if years_dict['I20_years'][i][j] >= 2007:
                            testing_ret.append(ret_dict['I20_returns'][i][j])
                    fo.write_csv_file(testing_ret, ret_out_dir)
                if i == 1:
                    ret_out_dir = os.path.join(out_dir, f'I20R20\\returns\\{ticker}_returns.csv')
                    for j in range(len(value[i])):
                        if years_dict['I20_years'][i][j] >= 2007:
                            testing_ret.append(ret_dict['I20_returns'][i][j])
                    fo.write_csv_file(testing_ret, ret_out_dir)
                if i == 2:
                    ret_out_dir = os.path.join(out_dir, f'I20R60\\returns\\{ticker}_returns.csv')
                    for j in range(len(value[i])):
                        if years_dict['I20_years'][i][j] >= 2007:
                            testing_ret.append(ret_dict['I20_returns'][i][j])
                    fo.write_csv_file(testing_ret, ret_out_dir)
        if "60" in key:
            for i in range(len(value)):
                testing_ret = []
                if i == 0:
                    ret_out_dir = os.path.join(out_dir, f'I60R5\\returns\\{ticker}_returns.csv')
                    for j in range(len(value[i])):
                        if years_dict['I60_years'][i][j] >= 2007:
                            testing_ret.append(ret_dict['I60_returns'][i][j])
                    fo.write_csv_file(testing_ret, ret_out_dir)
                if i == 1:
                    ret_out_dir = os.path.join(out_dir, f'I60R20\\returns\\{ticker}_returns.csv')
                    for j in range(len(value[i])):
                        if years_dict['I60_years'][i][j] >= 2007:
                            testing_ret.append(ret_dict['I60_returns'][i][j])
                    fo.write_csv_file(testing_ret, ret_out_dir)
                if i == 2:
                    ret_out_dir = os.path.join(out_dir, f'I60R60\\returns\\{ticker}_returns.csv')
                    for j in range(len(value[i])):
                        if years_dict['I60_years'][i][j] >= 2007:
                            testing_ret.append(ret_dict['I60_returns'][i][j])
                    fo.write_csv_file(testing_ret, ret_out_dir)
    print(f"{ticker} Returns Saved Successfully!")

## Generates all images for each stock in given data folder.
def generate_img_data(data_dir, out_dir):
    '''
    Creates all images for all stocks in given folder.
    Args:
        data_dir (string): The directory containing all OHLC stock prices.
        out_dir (string): The directory storing all OHLC images.
    Returns:
        None
    '''
    ## Keeps track of how many stock data have been loaded.
    counter = 1
    stock_num = len(os.listdir(data_dir))

    label_dict = {}
    years_dict = {}
    ret_dict = {}

    ## For each stock data, combines them into one dictionary.
    for filename in os.listdir(data_dir):
        ## Extract full file path and obtain stock data.
        file_path = os.path.join(data_dir, filename)
        ticker = os.path.splitext(os.path.basename(file_path))[0]
        stock_data = pd.read_csv(file_path)

        ## Converts the 'Date' column in the dataframe to a datetime object.
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])

        ## Creates temporary dictionaries for this particular stock data.
        data_dict, label_dict, years_dict, ret_dict = create_img_labels(stock_data)

        base_dict = create_training_img(data_dict, False, False)
        save_img(base_dict, label_dict, years_dict, ticker, 'base_img', out_dir)

        ma_dict = create_training_img(data_dict, False, True)
        save_img(ma_dict, label_dict, years_dict, ticker, 'ma_img', out_dir)

        vol_dict = create_training_img(data_dict, True, False)
        save_img(vol_dict, label_dict, years_dict, ticker, 'vol_img', out_dir)

        vol_ma_dict = create_training_img(data_dict, True, True)
        save_img(vol_ma_dict, label_dict, years_dict, ticker, 'vol_ma_img', out_dir)

        save_returns(ret_dict, years_dict, ticker, out_dir)

        ## Prints progress.
        print(f"{ticker} Stock {counter}/{stock_num} Completed!")
        counter = counter + 1

    print("All Images and Returns saved successfully!")

def main():
    data_dir = os.path.join('.', 'data', 'raw_dataset', 'stock_data')
    out_dir = os.path.join('.', 'data', 'processed_img', 'stock_img_test')
    generate_img_data(data_dir = data_dir, out_dir = out_dir)

if __name__ == "__main__":
    main()
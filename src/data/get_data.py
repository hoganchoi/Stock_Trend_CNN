## Acquires stock price data from all tickers in today's market. These stock price data's are adjusted and normalized. 
## The dataframe also contains the moving average as well. These dataframes are saved as csv files.

## Import necessary packages.
import yfinance as yf
import pandas as pd
import os

## Adjusts the rest of the stock data based on adjust close.
def adjust_stock_data(stock_data):
    '''
    Calculates the adjusted factor (split times dividend multiplier) and applies
    it to entire dataset (open, high, low, close).
    Args:
        stock_data (Dataframe): The dataframe acquired from yfinance.
    Returns:
        adj_data (Dataframe): The dataframe containing the adjusted values.
    '''
    adj_data = stock_data.copy()

    adj_factor = stock_data['Adj Close']/stock_data['Close']

    adj_data['Open'] = adj_factor * stock_data['Open']
    adj_data['High'] = adj_factor * stock_data['High']
    adj_data['Low'] = adj_factor * stock_data['Low']
    adj_data['Close'] = stock_data['Adj Close']

    adj_data = adj_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return adj_data

## Acquires OHLC chart from Yahoo Finance.
def get_stock_data(ticker, start_date, end_date):
    '''
    Create an adjusted dataframe of OHLC chart from Yahoo Finance.
    Args:
        ticker (string): The ticker representing firm name.
        start_date (string): The starting date (YYYY-MM-DD).
        end_date (string): The ending date (YYYY-MM-DD).
    Returns:
        final_data (Dataframe): An adjusted stock dataframe.
    '''
    data = yf.download(ticker, start_date, end_date)
    final_data = adjust_stock_data(data)
    final_data['Returns'] = final_data['Close'].pct_change()
    return final_data

## Normalizes the dataset based on returns.
def normalize_price(stock_data):
    '''
    Normalizes the data by setting first close price to 1.
    Args:
        stock_data (Dataframe): The dataframe containing stock price changes.
    Returns:
        norm_data (Dataframe): The dataframe containing normalized stock prices.
    '''
    norm_data = stock_data.copy()
    norm_data['Close Norm'] = 1

    ## Applies the equation presented in the paper.
    for i in range(len(stock_data) - 1):
        norm_data['Close Norm'][i + 1] = (1 + stock_data['Returns'][i + 1]) * norm_data['Close Norm'][i]

    norm_data['Open Norm'] = (stock_data['Open']/stock_data['Close']) * norm_data['Close Norm']
    norm_data['High Norm'] = (stock_data['High']/stock_data['Close']) * norm_data['Close Norm']
    norm_data['Low Norm'] = (stock_data['Low']/stock_data['Close']) * norm_data['Close Norm']

    return norm_data[['Open Norm', 'High Norm', 'Low Norm', 'Close Norm', 'Volume']]

## Adding the moving average column to dataframe.
def add_moving_average(stock_data):
    '''
    Calculates the Exponential-weighted Moving Average given stock data.
    Args:
        stock_data (Dataframe): The dataframe containing stock price changes.
    Returns:
        data (Dataframe): New dataframe with additional column containing
                          the moving average.
    '''
    data = stock_data.copy()
    data['Moving Avg'] = data['Close Norm'].ewm(alpha = 0.05).mean()
    return data

## Combine previous functions to generate an adjusted and preprocessed stock
## dataset.
def create_stock_data(ticker, start_date, end_date):
    '''
    Create adjusted and normalize stock dataset given parameters.
    Args:
        ticker (string): The ticker representing firm name.
        start_date (string): The starting date (YYYY-MM-DD).
        end_date (string): The ending date (YYYY-MM-DD).
    Returns:
        stock_data (Dataframe): An adjusted and normalized stock dataframe.
    '''
    stock_data = get_stock_data(ticker, start_date, end_date)
    stock_data = normalize_price(stock_data)
    stock_data = add_moving_average(stock_data)
    if stock_data.empty or not stock_data.index.year.isin([1993]).any():
        return pd.DataFrame()
    else:
        return stock_data

## Extract a list of tickers that were present on 1993 using URL.
def create_valid_tickers(url):
    '''
    Create a list of valid tickers from url.
    Args:
        url (string): An URL that contains all of firms on stock market today.
    Returns:
        valid_tickers (List): A list of valid tickers.
    '''
    tables = pd.read_html(url)
    tickers = tables[0]

    tickers['Date added'] = pd.to_datetime(tickers['Date added'], errors = 'coerce')
    tickers_filtered = tickers[tickers['Date added'] <= pd.Timestamp('1993-01-01')]

    valid_tickers = tickers_filtered['Symbol'].tolist()
    return valid_tickers

## Extract the list of tickers from csv file.
def create_valid_tickers_csv(csv_file):
    '''
    Given a csv file containing a list of tickers, obtain a list of tickers.
    Args:
        csv_file (string): A string representing the path to csv file.
    Returns:
        tickers (List): A list of tickers.
    '''
    tables = pd.read_csv(csv_file, header = None)
    tickers = tables[0].values.tolist()

    return tickers

## Generates stock dataframe for all tickers present in 1993 - 2019 using URL.
def generate_dataset_from_URL(in_dir, out_dir):
    '''
    Creates complete raw dataset.
    Args:
        in_dir (string): The URL hyperlink containing all the tickers on today's market.
        out_dir (string): The file directory storing all raw datasets.
    Returns:
        None
    '''
    valid_tickers = create_valid_tickers(url)
    start_date = '1993-01-01'
    end_date = '2019-12-31'

    for ticker in valid_tickers:
        csv_file_path = os.path.join(out_dir, f'{ticker}.csv')
        stock_df = create_stock_data(ticker, start_date, end_date)
        if not stock_df.empty:
            stock_df.to_csv(csv_file_path, index = True)
        else: 
            pass

    print("Created All Raw Datasets!")

## Generates stock dataframe for all tickers present in 1993 - 2019.
def generate_dataset(in_dir, out_dir):
    '''
    Creates complete raw dataset.
    Args:
        in_dir (string): The csv file path containing all the tickers from 1993 - 2019.
        out_dir (string): The file directory storing all raw datasets.
    Returns:
        None
    '''
    valid_tickers = create_valid_tickers_csv(in_dir)
    start_date = '1993-01-01'
    end_date = '2019-12-31'

    for ticker in valid_tickers:
        csv_file_path = os.path.join(out_dir, f'{ticker}.csv')
        stock_df = create_stock_data(ticker, start_date, end_date)
        if not stock_df.empty:
            stock_df.to_csv(csv_file_path, index = True)
        else: 
            pass

    print("Created All Raw Datasets!")

def main():
    in_dir = os.path.join('.', 'firm_list.csv')
    out_dir = os.path.join('.', 'data', 'raw_dataset', 'stock_data_test')
    generate_dataset(in_dir, out_dir)

if __name__ == "__main__":
    main()
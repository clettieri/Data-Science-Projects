# -*- coding: utf-8 -*-
"""
Created on Mon May 01 13:39:05 2017

@author: Chris
"""
import pandas as pd
import numpy as np

def get_data(symbol, start_date=None, end_date=None, with_features=True, n_bars_in_range=[1, 5, 20]):
    '''(string, string, string, bool, list) -> DataFrame
    
    Will return a dataframe from the file 'symbol.csv'
    of all columns from start to end date.
    
    Date expected as 'YYYY-MM-DD' or 'MM-DD-YYYY'
    Assums files located in 'data' folder.
    '''
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except:
        print "Error parsing start or end date"
    try:
        data = pd.read_csv('data/'+symbol+'.csv', index_col='date', parse_dates=True)
    except:
        print "Couldn't open " + symbol + ".csv"
    columns = ['adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']
    column_names = ['open', 'high', 'low', 'close', 'volume']
    data = data[columns]
    data.columns = column_names
    #Add features
    if with_features:
        data = add_features(data, n_bars_in_range)
    #Slice to dates
    output_df = data[start_date:end_date]
    return output_df

def add_features(df, n_bars_in_range):
    '''(DataFrame, list) -> DataFrame
    
    Will add feature columns to the data DF.
    '''
    #Daily Return (from yesterday close to today close)
    df['daily_ret'] = df['close'].pct_change() * 100
    #Return from day's open to day's close
    df['day_ret_from_open'] = (df['close'] - df['open'])/ df['open'] * 100
    #Opening Gap size
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['abs_gap'] = abs(df['gap'])
    #Add ATR
    df = add_atr_to_df(df, atr_period=20)
    #Add SMA
    df = add_sma_to_df(df)
    #Add close % in range
    for n in n_bars_in_range:
        df = add_close_percent_in_bar_range(df, n_bars=n)
      
    return df


'''================='''
'''Specific Features'''
'''================='''
def add_close_percent_in_bar_range(df, n_bars=1):
    '''(DataFrame, int) -> DataFrame
    
    Will create column for the % the bar close was in the
    range of the last n_bars (n_bar is todays bar).  The
    column will contain values from 0.0-1.0
    
    0.0 = closed at low of the day range
    1.0 = closed at high of the day range.
    
    This will be used to indicate strength of bar.
    '''
    #Get high and low for n_bar period
    df['highest_in_range'] = df['high'].rolling(n_bars).max()
    df['lowest_in_range'] = df['low'].rolling(n_bars).min()
    #Get size of bar
    df['bar_size'] = df['highest_in_range'] - df['lowest_in_range']
    #Close from low
    df['close_from_low'] = df['close'] - df['lowest_in_range']
    #Get in bar range
    df['in_bar_range_'+str(n_bars)] = df['close_from_low'] / df['bar_size']
    #Remove helper columns
    df.drop('bar_size', axis=1, inplace=True)
    df.drop('close_from_low', axis=1, inplace=True)
    df.drop('highest_in_range', axis=1, inplace=True)
    df.drop('lowest_in_range', axis=1, inplace=True)
    
    return df

def add_sma_to_df(df, sma_period=50):
    '''(DataFrame, int) -> DataFrame
    
    Will take a data frame of prices for a symbol and calculate
    a simple moving average based on the 'Adj Close'.  Returns
    a copy of the original DataFram with a new 'SMA' column.
    '''
    df['sma'] = df['close'].rolling(window=sma_period).mean()
    df['close_above_sma'] = np.where(df['close'] > df['sma'], 1, 0)
    return df

def add_atr_to_df(df, atr_period=20):
    '''(DataFrame, int) -> DataFrame
    
    Will take a data frame of prices for a symbol and calculate
    the ATR of two periods.  Returns a copy of the original DataFrame
    with two new columns 'ATR1', 'ATRX'.
    '''
    atr = df.copy()
    #Build potential True Range values using the 3 formulas
    atr['TR1'] = atr['high'] - atr['low']
    atr['TR2'] = abs(atr['high'] - atr['close'].shift(1))
    atr['TR3'] = abs(atr['low'] - atr['close'].shift(1))
    #Calculate the True Range
    def apply_func(tr1, tr2, tr3): 
        return max(tr1, tr2, tr3)
    atr['TrueRange'] = atr.apply(lambda row: apply_func(row['TR1'], row['TR2'], row['TR3']), axis=1)
    #Get the two ATR Values
    df['ATR1'] = atr['TR1'] #Yesterdays close not needed
    df['ATRX'] = atr['TrueRange'].rolling(window=atr_period).mean()
    #Add "big enough" if bar is bigger than average atr
    df['current_bar_bigger_than_avg'] = np.where(df['ATR1'] > 0.95*df['ATRX'], 1, 0)
    return df


if __name__ == "__main__":
    aapl = get_data('aapl', '2016-01-01')
    print aapl.head(10)

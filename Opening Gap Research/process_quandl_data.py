# -*- coding: utf-8 -*-
"""
Created on Mon May 01 12:50:13 2017

@author: Chris
"""
import pandas as pd

WIKI_DATA_PATH = 'data/wiki_prices.csv'

SYMBOLS = ['SPY', 'DIA', 'GLD', 'TLT', 'QQQ', 'USO', 'AAPL', 'NFLX', 'XOM', 'GS', 
           'JNJ', 'DIS', 'IBM', 'AMZN', 'TSLA']

def get_list_of_syms(df):
    '''(DataFrame) -> list
    
    Given a dataframe of price data, return list of 
    all unique symbols
    '''
    return df.ticker.unique()

def write_specific_stock_data_file(main_data_df, ticker):
    '''(DataFrame, string) -> writes a file
    
    Given the main dataframe and a symbol, will write the
    individual file for that stock.
    '''
    output_df = main_data_df[main_data_df['ticker'] == ticker]
    output_df.to_csv("data/"+ticker + ".csv")
    

def write_individual_stock_data_files(main_data_file_path, symbol_list=None):
    '''(string, list) -> writes files
    
    Will read the main data file (wiki from quandl) and
    will then write a new file for each indvidual symbol.
    If symbol_list is None then will write new file for
    all unique symbols in data set.
    '''
    #Read in main wiki data file
    main_data = pd.read_csv(main_data_file_path, index_col='date')
    unique_syms = get_list_of_syms(main_data)
    
    if symbol_list == None:
        for sym in unique_syms:
            write_specific_stock_data_file(main_data, sym)
    else:
        for sym in symbol_list:
            if sym not in unique_syms:
                print sym + " not in dataset"
                continue
            write_specific_stock_data_file(main_data, sym)
            
if __name__ == "__main__":
    write_individual_stock_data_files(WIKI_DATA_PATH, SYMBOLS)
    print "Done"

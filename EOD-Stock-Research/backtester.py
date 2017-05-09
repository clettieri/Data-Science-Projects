'''
Given a dataframe of daily returns and predicted trades,
this will output certain metrics as if the ml signals were taken
with the close price as entry.
'''

import pandas as pd

def get_total_return(df):
    '''(DataFrame) -> float 
    
    Given a dataframe of predictions will get the cumsum()
    of daily returns based on whether or not the trade was
    in the same direction. 
    +daily return if same side, -daily return if not
    
    Expects a column of 'predictions' where:
        -1 = short
        0 = flat
        1 = long
    '''
    #Print trade count - TODO - put all this in print output function
    #print df['predictions'].value_counts()
    #Return = 1 * return (if long) or -1 * return if short 
    #a -1 short * -1 return = positive trade return    
    df['trade_return'] = df['predictions'] * df['next_day_return']
    total_return = df['trade_return'].cumsum()[-1]
    
    #TODO Plot trade_return cumsum
    counts_df = df[df['predictions'] != 0]
    print df['predictions'].value_counts()
    total_trades = len(counts_df)
    return_per_trade = total_return / total_trades
    print ""
    print "TotalReturn:%f  #Trades:%d  ReturnPerTrade:%f " % (total_return, total_trades, return_per_trade)
    
    
    


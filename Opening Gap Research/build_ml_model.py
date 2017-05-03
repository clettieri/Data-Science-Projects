"""
After doing with exploratory data analyis, use this script to prepare the
data for building a machine learning model.

-Get values for features and get label values into arrays.
-Split data into a training and testing set
-Finally train/test model using time series split 
"""
import numpy as np
import pandas as pd
from get_data import get_data
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

'''
FEATURE_LIST = ['day_ret_from_open', 'gap', 'abs_gap', 'ATR1', 'ATRX',
                'current_bar_bigger_than_avg', 'close_above_sma', 
                'in_bar_range_1', 'in_bar_range_5', 'in_bar_range_20']
'''
FEATURE_LIST = ['day_ret_from_open', 'gap', 'ATR1', 'ATRX',
                'in_bar_range_1', 'in_bar_range_5', 'in_bar_range_20']

LABEL_COL = ['next_day_direction']
#LABEL_COL = ['next_day_return']

'''==========================='''
'''Pre-Process the data for ML'''
'''==========================='''
def add_label_cols_to_df(df):
    '''(DataFrame) -> DataFrame
    
    Will create several columns that are potential target variables
    to test for.  
    
    'next_day_return' -> float value, continous
    'next_day_direction' -> bool value, discrete (1=up, 0=flat, -1=down)
        --flat = abs(tomorrows_return) <= 0.3%
    '''
    #Next Day Return
    df['next_day_return'] = df['daily_ret'].shift(-1)
    
    #Next Day Direction
    def next_day_direction(row):
        if row['next_day_return'] > 0.75:
            return 1
        elif row['next_day_return'] < -0.75:
            return -1
        else:
            return 0
    df['next_day_direction'] = df.apply(next_day_direction, axis=1)
    #Slice off last row since won't have label for it
    return df[:-1]

def get_train_and_test_sets(df, feature_cols=FEATURE_LIST, label_col=LABEL_COL, 
                            percent_test_set=0.3):
    '''(DataFrame, list, string, float) -> X_train, X_test, Y_train, Y_test
    
    Given a data frame, list of feature column names, a label column name,
    and a percentage of data to be retained as test set.  This function
    will split the data before doing cross validation.  Cross-validation
    and training should be done on X_train, and X_test
    '''
    end_of_train_index = int(len(df) * (1-percent_test_set))
    #Get X values
    X_train = df[feature_cols][:end_of_train_index].values
    X_test = df[feature_cols][end_of_train_index:].values
    #Get Y values
    Y_train = df[label_col][:end_of_train_index].values
    Y_test = df[label_col][end_of_train_index:].values     
    #assert(len(df) == (len(Y_train) +len(Y_test)))
    return X_train, X_test, Y_train, Y_test

def run_time_series_cross_validation(classifer, X_train, Y_train, n_splits=5, random_state=7):
    '''(sklearn classifierd, array, array, int, int) -> list of floats (scores)
    
    This will act as a test-harness to split the time series data and
    run cross val score using our model.  Will return the list of
    scores as well as print the mean.
    '''
    cv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(classifer, X_train, Y_train, cv=cv, scoring='accuracy')
    print "Avg. Score: " + str(scores.mean())
    print "Min Score: " + str(scores.min())
    print "Max Score: " + str(scores.max())
    return scores

def main(symbol, start_date, run_cv=True, run_test=False):
    '''(string, string) -> None
    
    This will get data, process data, and run a cross-val
    using the ML model specified.
    '''
    #Get data and features first from get_data.py
    df = get_data(symbol, start_date)
    #Add label columns
    df = add_label_cols_to_df(df)
    #Split
    X_train, X_test, Y_train, Y_test = get_train_and_test_sets(df)
    #Initiate ML algo, run cross validation
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=50)
    if run_cv:
        scores = run_time_series_cross_validation(clf, X_train, Y_train, n_splits=5)
    #Train & Test the classifier
    if run_test:
        clf.fit(X_train, Y_train)
        predictions = pd.Series(clf.predict(X_test))
        print predictions.value_counts()
        print "Model Accuracy: " + str(accuracy_score(Y_test, predictions))
        importances = pd.DataFrame(zip(clf.feature_importances_, FEATURE_LIST), columns=['Importance', 'Name'])
        print importances.sort_values('Importance', ascending=False)
    


if __name__ == "__main__":
    main('aapl', '2000-01-01', run_cv=True, run_test=True)

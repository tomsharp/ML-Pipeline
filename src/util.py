import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample


def dq_range_checks(df, ranges):
    '''
    Description: This function checks whether the min and max value of each column are within the acceptable range. 
    Input: 
        df [Pandas DataFrame]: DataFrame containing date for training/testing model
        ranges [dictionary]: a dictionary containing a column name (key) and a tuple (value) structured as (range_min, range_max)
    Output: None
    '''
    for k,v in ranges.items():
        assert df[k].min() >= v[0], \
            print("Data Quality Check Failed: min value out of range for '{}' column".format(k))
        assert df[k].max() <= v[1], \
            print("Data Quality Check Failed: max value out of range for '{}' column".format(k))

def dq_set_checks(df, sets):
    '''
    Description: This function checks whether the unique values of each column are as specified 
    Input: 
        df [Pandas DataFrame]: DataFrame containing date for training/testing model
        sets [dictionary]: a dictionary containing column names (key) and a list of unique values for that column (value)
    Output: None
    '''
    for k,v in sets.items():
        l = list(df[k].fillna('').unique())
        l.sort()
        v.sort()
        assert l == v, "Data Quality Check Failed: set of unique values for {} is not {}".format(k,v)


def evaluate_model(predict_probs, threshold, y_test):
    preds = [1 if prob >= threshold else 0 for prob in predict_probs]
    f1 = f1_score(y_test, preds)
    (tn, fp, fn, tp) = confusion_matrix(y_test, preds).ravel()
    metrics = {'f1': f1, 
               'tn': tn,'fp': fp, 'fn': fn, 'tp': tp}
    return metrics
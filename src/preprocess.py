import pandas as pd 
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from config import DATA_PATH, ranges, sets, risk_encoding, savings_encodings, checking_encodings
from util import dq_range_checks, dq_set_checks


def prep_data(df):
    ########################
    # Data Quality checks 
    ########################
    # DQ Check: Columns 
    cols = ['age', 'gender', 'job_cat', 'housing', 'savings',
        'checking', 'loan_amount', 'duration', 'purpose', 'risk',
        'customer_loyalty']
    assert sum([c in df.columns for c in cols]) == len(cols), "Data Quality Check Failed: Missing expected column(s)"
    df = df[cols]

    # DQ Check: Ranges
    dq_range_checks(df, ranges)

    # DQ Check: Sets
    dq_set_checks(df, sets)


    ########################
    # Encode variables
    ########################
    # encode ordinal variables
    df['risk'] = df['risk'].apply(lambda x: risk_encoding[x])
    df['savings'] = df['savings'].fillna('none').apply(lambda x: savings_encodings[x])
    df['checking'] = df['checking'].fillna('none').apply(lambda x: checking_encodings[x])

    # encode categorical variables
    job_dummies = pd.get_dummies(df['job_cat'], prefix='jobcat')
    job_dummies.drop('jobcat_0', axis=1, inplace=True)
    assert list(job_dummies.columns)==['jobcat_1', 'jobcat_2', 'jobcat_3']

    housing_dummies = pd.get_dummies(df['housing'], prefix='housing')
    housing_dummies.drop('housing_free', axis=1, inplace=True)
    assert list(housing_dummies.columns)==['housing_own', 'housing_rent']

    purpose_dummies = pd.get_dummies(df['purpose'], prefix='purpose')
    purpose_dummies.drop('purpose_vacation/others', axis=1, inplace=True)
    assert list(purpose_dummies.columns) == ['purpose_business', 'purpose_car', 
                                            'purpose_domestic appliances', 'purpose_education', 
                                            'purpose_furniture/equipment', 'purpose_radio/TV',
                                            'purpose_repairs']

    # merge categorical econdings
    df = df[['age', 'savings', 'checking', 'loan_amount', 'duration', 'risk', 'customer_loyalty']]
    for d in [job_dummies, housing_dummies, purpose_dummies]:
        df = df.merge(d, left_index=True, right_index=True)

    ################################
    # Drop rows with missing vals
    ################################
    df = df[df.isna().sum(axis=1)==0]

    return df 



def split_data(df, target, test_size=0.25, balance_classes=True):
    features = [c for c in df.columns if c!=target]

    if balance_classes == True: 
        n_samples = min(df[df[target]==0][target].count(), df[df[target]==1][target].count())
        target0_samples = resample(df[df[target]==0], replace=True, n_samples=n_samples, random_state=0)
        target1_samples = resample(df[df[target]==1], replace=True, n_samples=n_samples, random_state=0)
        df = pd.concat([target0_samples, target1_samples])

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    return X_train, X_test, y_train, y_test

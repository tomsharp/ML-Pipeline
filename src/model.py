import json
import pandas as pd 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocess import prep_data, split_data
from config import DATA_PATH
from util import evaluate_model


gs_scoring_critera = 'f1'
test_thresholds = True

models = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=0),
        'param_grid': {
            'penalty': ['none'],
            'solver': ['saga']
        },
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=0),
        'param_grid': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best'],
            'max_depth': [2, 5, 10, 15],
            'min_samples_split': [2, 5, 10, 15]
        },
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=0),
        'param_grid': {
            'n_estimators': [10, 100, 1000],
            'criterion': ['gini'],
            'max_depth': [2, 5, 10, 15],
            'min_samples_split': [2, 5, 10, 15]
        },
    },
}


if __name__ == '__main__':
    # read in, prepare, and split data
    print('\n\n-----------Prepping data-----------\n')
    df = pd.read_csv(DATA_PATH)
    df = prep_data(df)
    X_train, X_test, y_train, y_test = split_data(df, target='risk')


    print('\n\n-----------Fitting Models-----------\n')
    results=[]
    for k in models.keys():
        print('\nFitting and testing {}'.format(k))

        # instantiate gridsearch
        gcv = GridSearchCV(
            estimator = models[k]['model'],
            param_grid = models[k]['param_grid'],
            scoring=gs_scoring_critera,
            cv=5
        )

        # run gridsearch, fit best estimator
        gcv.fit(X_train, y_train)
        model = gcv.best_estimator_
        model.fit(X_train, y_train)

        # evaluate
        d={}
        probs = model.predict_proba(X_test)
        probs_positive = [p[1] for p in probs]

        if test_thresholds==True:
            for threshold in np.linspace(0.01, 0.99, 50):
                d = evaluate_model(probs_positive, threshold, y_test)
                d['threshold'] = threshold
                d['model'] = k
                d['params'] = gcv.best_params_
                results.append(d)
        else:
            threshold = 0.5
            d = evaluate_model(probs_positive, threshold, y_test)
            d['threshold'] = threshold
            d['model'] = k
            d['params'] = gcv.best_params_
            results.append(d)


    print('\n\n-----------Results-----------\n')
    # print best result
    print('Best result(s):')
    results = pd.DataFrame(results)
    best_result = results[results['f1']==results['f1'].max()]
    print(best_result[['f1', 'threshold', 'model', 'params']])

    # save to csv
    print('\nSaving all results')
    results.to_csv('modeling/model_results.csv', index=False)

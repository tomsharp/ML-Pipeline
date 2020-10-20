# csv path
DATA_PATH = 'data_sets/Credit risk assessment.csv'

# data columns 
cols = ['age', 'gender', 'job_cat', 'housing', 'savings',
       'checking', 'loan_amount', 'duration', 'purpose', 'risk',
       'customer_loyalty']

    
# ranges, sets for numeric, categorical data 
ranges = {
    'age': (18, 100),
    'loan_amount': (100, 20000),
    'duration': (3, 72),
    'customer_loyalty': (0, 1.0)
}
sets = {
    'gender': ['female', 'male', 'unknown'],
    'job_cat': [0, 1, 2, 3],
    'housing': ['free', 'own', 'rent'],
    'savings': ['', 'high', 'low', 'moderate', 'very_high'],
    'checking': ['', 'high', 'low', 'moderate'],
    'purpose': ['business', 'car', 'domestic appliances', 'education', 
                'furniture/equipment', 'radio/TV', 'repairs', 'vacation/others'],
    'risk': ['bad', 'good']
}


# categorical encoding maps
risk_encoding = {
    'good':0,
    'bad':1
}
savings_encodings = {
    'none':0,
    'low':1,
    'moderate':2,
    'high':3,
    'very_high':4
}
checking_encodings = {
    'none':0,
    'low':1,
    'moderate':2,
    'high':3
}
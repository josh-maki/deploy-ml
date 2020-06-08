import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    #data = pd.read_csv('titanic.csv')
    data = pd.read_csv(df_path)
    return data



def divide_train_test(df, target):
    # Function divides data set in train and test
    vars_num = [c for c in df.columns if df[c].dtypes!='O' and c!=target]
    vars_cat = [c for c in df.columns if df[c].dtypes=='O']
    X_train, X_test, y_train, y_test = train_test_split(
    df.drop(target, axis=1),  # predictors
    df[target],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)
    return X_train, X_test, y_train, y_test
    



def extract_cabin_letter(df, var):
    # captures the first letter
    df['cabin'] = df['cabin'].str[0] # captures the first letter
    CabinLetters = df['cabin'].unique()
    return df



def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    return df


    
def impute_na(df, var, IMPUTATION_DICT):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    replacement_val = IMPUTATION_DICT[var]
    df[var].fillna(replacement_val, inplace=True)
    return df



def remove_rare_labels(df, FREQUENT_LABELS, var):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    frequent_ls = FREQUENT_LABELS[var]
    df[var] = np.where(df[var].isin(frequent_ls), df[var], 'Rare')
    return df



def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    df = df.copy()
    tempTrainDummies = pd.get_dummies(df[var], prefix=var, drop_first=True)
    #print(var+':', tempTrainDummies.columns.tolist())
    df = pd.concat([df,tempTrainDummies], axis=1)
    df.drop(labels=[var], axis=1, inplace=True)
    return df



def check_dummy_variables(df, DUMMY_VARIABLES):
    
    # check that all missing variables were added when encoding, otherwise
    # add the ones that are missing
    for key in DUMMY_VARIABLES.keys():
        col_list = DUMMY_VARIABLES[key]
        for col in col_list:
            if col not in df.columns:
                print('missing '+col)
                df[col] = 0
    return df
    

def train_scaler(df, ordered_columns, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df[ordered_columns])
    joblib.dump(scaler, output_path) 
    return scaler
  
    

def scale_features(df, ordered_columns, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    df[ordered_columns] = scaler.transform(df[ordered_columns])
    return df



def train_model(df, ordered_columns, target, output_path):
    # train and save model
    model = LogisticRegression(C=0.0005, random_state=0)
    model.fit(df[ordered_columns], target)
    joblib.dump(model, output_path)
    return model



def predict(df, ordered_columns, output_path):
    # load model and get predictions
    model = joblib.load(output_path)
    yhat = model.predict(df[ordered_columns])
    return yhat

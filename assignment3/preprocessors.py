import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables


    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        self.na_cols = X.columns[X.isnull().any()]
        return self


    def transform(self, X):
        # add indicator
        X = X.copy()
        for var in self.na_cols:
            X[var+'_NA'] = np.where(X[var].isnull(), 1, 0)
        return X


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        self.imputer_dict_ = {}
        for var in self.variables:
            self.imputer_dict_[var] = 'missing'
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.imputer_dict_[var], inplace=True)
        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for var in self.variables:
            self.imputer_dict_[var] = X[var].mode()[0]
        return self

    def transform(self, X):
        #replace missing values with mode
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.imputer_dict_[var], inplace=True)
        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        for var in self.variables:
            X[var] = X[var].str[0]
            #X['cabin'] = X['cabin'].str[0]
        return X

# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.tol = tol

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}
        for var in self.variables:
            t = pd.Series(X[var].value_counts()/np.float(len(X)))
            self.encoder_dict_[var] = list(t[t >= self.tol].index)
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.encoder_dict_[var]), X[var], 'rare')
        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def fit(self, X, y=None):

        # HINT: persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        # get dummies
        for var in self.variables:
            dummies = pd.get_dummies(X[var], prefix=var, drop_first=True)
            X=pd.concat([X,dummies], axis=1)
        # drop original variables
        X.drop(labels=self.variables, axis=1, inplace=True)
        
        #drop extra one-hot-encoded variables
        self.extraEncodings = [col for col in dummies.columns if col not in self.dummies]
        X.drop(labels = self.extraEncodings, axis = 1, inplace = True)
        
        # add missing dummies if any
        self.missingEncodings = [col for col in self.dummies if col not in dummies.columns]
        X.drop(labels = self.missingEncodings, axis = 1, inplace = True)

        return X

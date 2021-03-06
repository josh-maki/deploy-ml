from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [
        ('missing_indicator', pp.MissingIndicator(variables = config.CATEGORICAL_VARS+config.NUMERICAL_VARS)),
        ('categorical_imputer', pp.CategoricalImputer(variables = config.CATEGORICAL_VARS)),
        ('numerical_imputer', pp.NumericalImputer(variables = config.NUMERICAL_VARS)),
        ('extract_first_letter', pp.ExtractFirstLetter(variables = ['cabin',])),
        ('rare_label_categorical_encoder', pp.RareLabelCategoricalEncoder(tol=0.05, variables = config.CATEGORICAL_VARS)),
        ('categorical_encoder', pp.CategoricalEncoder(variables = config.CATEGORICAL_VARS))
    ]
    )
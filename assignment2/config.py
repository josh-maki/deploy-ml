# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===============

# imputation parameters
IMPUTATION_DICT = {'age': 28.0, 'fare':14.452, 'sex': 'Missing', 'cabin': 'Missing', 'embarked': 'Missing', 'title': 'Missing'}


# encoding parameters
FREQUENT_LABELS = {'sex': ['female', 'male'], 'cabin': ['C', 'Missing', 'Rare'], 'embarked': ['C', 'Q', 'S'], 'title': ['Miss', 'Mr', 'Mrs', 'Rare']}


DUMMY_VARIABLES = {'sex': ['sex_male'], 'cabin': ['cabin_Missing', 'cabin_Rare'], 'embarked': ['embarked_Q', 'embarked_Rare', 'embarked_S'], 'title': ['title_Mr', 'title_Mrs', 'title_Rare']}




# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_TO_IMPUTE = ['pclass', 'age', 'sibsp', 'parch', 'fare']

ORDERED_COLUMNS = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'age_NA', 'fare_NA', 'sex_male', 'cabin_Missing', 'cabin_Rare', 'embarked_Q', 'embarked_Rare', 'embarked_S', 'title_Mr', 'title_Mrs', 'title_Rare']
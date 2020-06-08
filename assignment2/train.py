import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
df = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(df, config.TARGET)

# get first letter from cabin variable
X_train = pf.extract_cabin_letter(X_train, 'cabin')


# impute categorical variables
for var in config.CATEGORICAL_VARS:
    X_train = pf.impute_na(X_train, var, config.IMPUTATION_DICT)


# impute numerical variable
# since the notebook just uses age and fare, we will ignore the "NUMERICAL TO IMPUTE"
for var in ['age', 'fare']:
    X_train = pf.impute_na(X_train, var, config.IMPUTATION_DICT)


# add missing indicator #Note that I added this to conform train.py with notebook.
for var in ['age', 'fare']:
    X_train = pf.add_missing_indicator(X_train, var)


# Group rare labels
for var in config.CATEGORICAL_VARS:
    X_train = pf.remove_rare_labels(X_train, config.FREQUENT_LABELS, var)


# encode categorical variables
for var in config.CATEGORICAL_VARS:
    X_train = pf.encode_categorical(X_train, var)


# check all dummies were added
X_train = pf.check_dummy_variables(X_train, config.DUMMY_VARIABLES)

# train scaler and save
pf.train_scaler(X_train, config.ORDERED_COLUMNS, config.OUTPUT_SCALER_PATH)


# scale train set
X_train = pf.scale_features(X_train, config.ORDERED_COLUMNS, config.OUTPUT_SCALER_PATH)


# train model and save
pf.train_model(X_train, config.ORDERED_COLUMNS, y_train, config.OUTPUT_MODEL_PATH)


print('Finished training')
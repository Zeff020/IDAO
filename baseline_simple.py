
import os
import pandas as pd
import numpy as np
#import xgboost
import catboost
import utils
import scoring
from sklearn.model_selection import train_test_split

def relu(x):
  return np.array([ (i>0) * abs(i) for i in x ])


DATA_PATH = "./data"

train, test = utils.load_small_data_csv(
    DATA_PATH, "train_small.csv", "test_small.csv", utils.SIMPLE_FEATURE_COLUMNS)

print(train.head())
print(test.head())


# first train on part of the training sample, and use a subset to validate
train_part, validation = train_test_split(train, test_size=0.25, shuffle=True)

model = catboost.CatBoostClassifier(iterations=2, depth=2, learning_rate=1, loss_function="Logloss",logging_level="Verbose")

model.fit(train_part.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values,
          train_part.label.values,
          sample_weight=relu(train_part.weight.values),  ## deal with negative weights...
          verbose=True)

# make predictions on validation part
validation_predictions = model.predict_proba(validation.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values)[:, 1]

# test the Yandex-defined scoring performance
result = scoring.rejection90(validation.label.values, validation_predictions, sample_weight=validation.weight.values)
print(result)

# now train on whole training set, for optimal learning
model.fit(train.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values, 
    train.label, 
    sample_weight=relu(train.weight))

model.save_model("baseline_simple_model.xgb")

# make predictions for the test set
predictions = model.predict_proba(test.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values)[:, 1]

# write out as csv of 'index, prediction'
pd.DataFrame(data={"prediction": predictions}, index=test.index).to_csv(
    "baseline_simple_submission.csv", index_label=utils.ID_COLUMN)



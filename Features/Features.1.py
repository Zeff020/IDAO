import utils
import numpy as np
import pandas as pd
from mathutils.geometry import intersect_point_line
from numpy.linalg import norm

print("Loading in data...")
# Loading data sets
PATH = r"./data/"
train, test = utils.load_data_csv(PATH, utils.SIMPLE_FEATURE_COLUMNS)
print("Finished loading in data, starting feature generation for train dataset")

# Create features for train dataset
PointResiduals, Angles, LineSlope1, LineSlope2, LineSlope3, FirstPointResiduals, FourthPointResiduals = utils.kink(train)

print("Finished feature generation, adding features to train dataframe")
# Add features for train dataframe
train['PointResiduals'] = pd.Series(PointResiduals, index=train.index)
train['Angles'] = pd.Series(Angles, index=train.index)
train['LineSlope1'] = pd.Series(LineSlope1, index=train.index)
train['LineSlope2'] = pd.Series(LineSlope2, index=train.index)
train['LineSlope3'] = pd.Series(LineSlope3, index=train.index)
train['FirstPointResiduals'] = pd.Series(FirstPointResiduals, index=train.index)
train['FourthPointResiduals'] = pd.Series(FourthPointResiduals, index=train.index)

print("Finished adding features to train dataframe, starting feature generation for test dataset")
# Create features for test dataset
PointResiduals, Angles, LineSlope1, LineSlope2, LineSlope3, FirstPointResiduals, FourthPointResiduals = utils.kink(test)
print("Finished feature generation for test dataset, adding features to test dataframe")
# Adding some columns to the test dataframe
test['PointResiduals'] = pd.Series(PointResiduals, index=test.index)
test['Angles'] = pd.Series(Angles, index=test.index)
test['LineSlope1'] = pd.Series(LineSlope1, index=test.index)
test['LineSlope2'] = pd.Series(LineSlope2, index=test.index)
test['LineSlope3'] = pd.Series(LineSlope3, index=test.index)
test['FirstPointResiduals'] = pd.Series(FirstPointResiduals, index=test.index)
test['FourthPointResiduals'] = pd.Series(FourthPointResiduals, index=test.index)

print("Finished generating and adding data, saving to csv")
# Saving the resulting datasets
train.to_csv(r"./data" + r"/trainMoreFeatures" + ".csv.gz", compression = 'gzip')
test.to_csv(r"./data" + r"/testMoreFeatures" + ".csv.gz", compression = 'gzip')

print("Finished everything :)")
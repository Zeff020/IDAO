import utils
import numpy as np
import pandas as pd
from mathutils.geometry import intersect_point_line
from numpy.linalg import norm


# Loading data sets
PATH = r"./data/"
train, test = utils.load_data_csv(PATH, utils.SIMPLE_FEATURE_COLUMNS)

# Create features for train dataset
PointResiduals, Angles, LineSlope, FirstPointResiduals, FourthPointResiduals = utils.kink(train)

# Add features for train dataframe
train['PointResiduals'] = pd.Series(PointResiduals, index=train.index)
train['Angles'] = pd.Series(Angles, index=train.index)
train['LineSlope'] = pd.Series(LineSlope, index=train.index)
train['FirstPointResiduals'] = pd.Series(FirstPointResiduals, index=train.index)
train['FourthPointResiduals'] = pd.Series(FourthPointResiduals, index=train.index)

# Create features for test dataset
PointResiduals, Angles, LineSlope, FirstPointResiduals, FourthPointResiduals = utils.kink(test)

# Adding some columns to the test dataframe
test['PointResiduals'] = pd.Series(PointResiduals, index=test.index)
test['Angles'] = pd.Series(Angles, index=test.index)
test['LineSlope'] = pd.Series(LineSlope, index=test.index)
test['FirstPointResiduals'] = pd.Series(FirstPointResiduals, index=test.index)
test['FourthPointResiduals'] = pd.Series(FourthPointResiduals, index=test.index)

# Saving the resulting datasets
train.to_csv(r"./data" + r"/trainMoreFeatures" + ".csv.gz", compression = 'gzip')
test.to_csv(r"./data" + r"/testMoreFeatures" + ".csv.gz", compression = 'gzip')
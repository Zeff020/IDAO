import utils
import numpy as np
import pandas as pd
from mathutils.geometry import intersect_point_line
from numpy.linalg import norm

print("Loading in data...")
# Loading data sets
PATH = r"./data/"
train, test = utils.load_small_data_csv(PATH,"train_smaller100.csv.gz","test_smaller100.csv.gz", utils.SIMPLE_FEATURE_COLUMNS)
print("Finished loading in data, starting feature generation for train dataset")
print("Starting kink")
#### TRAIN FEATURES ####

DataSet = train
Location_info = DataSet.loc[: , "MatchedHit_X[0]":"MatchedHit_Z[3]"]
countertrain = 0
countertest = 0

PointResiduals = np.array([])
Angles = np.array([])
LineSlope1 = np.array([])
LineSlope2 = np.array([])
LineSlope3 = np.array([])
FourthPointResiduals = np.array([])
FirstPointResiduals = np.array([])
Shape = DataSet.shape[0]

train['PointResiduals'] = pd.Series(PointResiduals)
train['Angles'] = pd.Series(Angles)
train['LineSlope1'] = pd.Series(LineSlope1)
train['LineSlope2'] = pd.Series(LineSlope2)
train['LineSlope3'] = pd.Series(LineSlope3)
train['FirstPointResiduals'] = pd.Series(FirstPointResiduals)
train['FourthPointResiduals'] = pd.Series(FourthPointResiduals)

train.to_csv(r"./data" + r"/trainMoreFeatures" + ".csv")


for i in range(0,DataSet.shape[0]):
    
    ResidualsSize = 0

    if i % 10000 == 0:
        print(" Evaluated " + str(i) + " features out of a total of " + str(Shape)) 

    LineSlope = np.array([])
    # Extracting info on the i-th particle's coordinates
    Particle_Path_Points = Location_info.loc[i,:]
    X = Particle_Path_Points.loc['MatchedHit_X[0]':'MatchedHit_X[3]'].values
    Y = Particle_Path_Points.loc['MatchedHit_Y[0]':'MatchedHit_Y[3]'].values
    Z = Particle_Path_Points.loc['MatchedHit_Z[0]':'MatchedHit_Z[3]'].values

    data = np.concatenate((X[:, np.newaxis], 
                        Y[:, np.newaxis], 
                        Z[:, np.newaxis]), 
                        axis=1)

    datamean = data.mean(axis=0)

    # uu, dd and vv contain information on the fit. In fact, vv[0] contains the direction of the best fit (least squares)
    uu, dd, vv = np.linalg.svd(data - datamean)

    # Best fit line with length between -2500 and 2500 with 2 datapoints (it's a straight line so that's enough)
    linepts = vv[0] * np.mgrid[-2500:2500:2j][:, np.newaxis]
    LineSlope = vv[0]
    LineSlope1 = np.append(LineSlope1, LineSlope[0])
    LineSlope2 = np.append(LineSlope2, LineSlope[1])
    LineSlope3 = np.append(LineSlope3, LineSlope[2])


    # Shift by the mean to get the line in the right place (centered)
    linepts += datamean

    # Do a little loop where we add the residuals of all four of the points
    for j in range (0,4):
        intersect = intersect_point_line(data[j], linepts[0], linepts[1])
        ResidualsSize += abs(sum(data[j] - intersect[0]))

    PointResiduals = np.append(PointResiduals, ResidualsSize)

    # Singular value decomposition on first line to get its derivative (angle)
    DataFirstTwo = data[0:2,:]
    datamean = DataFirstTwo.mean(axis=0)
    uu, dd, vv = np.linalg.svd(DataFirstTwo - datamean)
    FirstLineAngle = vv[0]
    
    
    # Singular value decomposition on second line to get its derivative (angle)
    DataSecondTwo = data[2:4,:]    
    datamean = DataSecondTwo.mean(axis=0)
    uu, dd, vv = np.linalg.svd(DataSecondTwo - datamean)
    SecondLineAngle = vv[0]
    
    # Finding the angle between the vectors made up by the first and second line. Simply dot product as both lines are normalized (norm=1 for both)
    Angle = np.dot(FirstLineAngle,SecondLineAngle)
    Angles = np.append(Angles, Angle)

    # Singular value decomposition for first three points to get residuals with the fourth point
    DataFirstThree = data[0:3,:]
    datamean = DataFirstThree.mean(axis=0)
    uu, dd, vv = np.linalg.svd(DataFirstThree - datamean)
    
    linepts = vv[0] * np.mgrid[-2500:2500:2j][:, np.newaxis]
    linepts += datamean
    
    intersect = intersect_point_line(data[3], linepts[0], linepts[1])
    ResidualSize = abs(norm(data[3] - intersect[0]))
    
    FourthPointResiduals = np.append(FourthPointResiduals, ResidualSize)
    
    # Singular value decomposition for last three points to get residuals with the first point
    DataLastThree = data[1:4,:]
    datamean = DataLastThree.mean(axis=0)
    uu, dd, vv = np.linalg.svd(DataLastThree - datamean)
    
    linepts = vv[0] * np.mgrid[-2500:2500:2j][:, np.newaxis]
    linepts += datamean
    
    intersect = intersect_point_line(data[0], linepts[0], linepts[1])
    ResidualSize = abs(norm(data[0] - intersect[0]))
    
    FirstPointResiduals = np.append(FirstPointResiduals, ResidualSize)

    if i % 100000 == 0:
        df = train.loc[i-100000:i,:]
        df['PointResiduals'] = pd.Series(PointResiduals)
        df['Angles'] = pd.Series(Angles)
        df['LineSlope1'] = pd.Series(LineSlope1)
        df['LineSlope2'] = pd.Series(LineSlope2)
        df['LineSlope3'] = pd.Series(LineSlope3)
        df['FirstPointResiduals'] = pd.Series(FirstPointResiduals)
        df['FourthPointResiduals'] = pd.Series(FourthPointResiduals)

        with open('./data/trainMoreFeatures.csv', 'a') as f:
            df.to_csv(f, header=False)

        countertrain += 1

    if i == DataSet.shape[0]-1:
        df = pd.DataFrame([],columns = ["PointResiduals", "Angles", "LineSlope1", "LineSlope2", "LineSlope3", "FirstPointResiduals", "FourthPointResiduals"])
        df['PointResiduals'] = pd.Series(PointResiduals)
        df['Angles'] = pd.Series(Angles)
        df['LineSlope1'] = pd.Series(LineSlope1)
        df['LineSlope2'] = pd.Series(LineSlope2)
        df['LineSlope3'] = pd.Series(LineSlope3)
        df['FirstPointResiduals'] = pd.Series(FirstPointResiduals)
        df['FourthPointResiduals'] = pd.Series(FourthPointResiduals)

        with open('./data/trainMoreFeatures.csv', 'a') as f:
            df.to_csv(f, header=False)

        df = pd.DataFrame([])
        PointResiduals = np.array([])
        Angles = np.array([])
        LineSlope1 = np.array([])
        LineSlope2 = np.array([])
        LineSlope3 = np.array([])
        FourthPointResiduals = np.array([])
        FirstPointResiduals = np.array([])
#### TEST FEATURES ####

DataSet = test
Location_info = DataSet.loc[: , "MatchedHit_X[0]":"MatchedHit_Z[3]"]


PointResiduals = np.array([])
Angles = np.array([])
LineSlope1 = np.array([])
LineSlope2 = np.array([])
LineSlope3 = np.array([])
FourthPointResiduals = np.array([])
FirstPointResiduals = np.array([])
Shape = DataSet.shape[0]

test['PointResiduals'] = pd.Series(PointResiduals)
test['Angles'] = pd.Series(Angles)
test['LineSlope1'] = pd.Series(LineSlope1)
test['LineSlope2'] = pd.Series(LineSlope2)
test['LineSlope3'] = pd.Series(LineSlope3)
test['FirstPointResiduals'] = pd.Series(FirstPointResiduals)
test['FourthPointResiduals'] = pd.Series(FourthPointResiduals)

test.to_csv(r"./data" + r"/testMoreFeatures" + ".csv")


for i in range(0,DataSet.shape[0]):
    
    ResidualsSize = 0

    if i % 10000 == 0:
        print(" Evaluated " + str(i) + " features out of a total of " + str(Shape)) 

    LineSlope = np.array([])
    # Extracting info on the i-th particle's coordinates
    Particle_Path_Points = Location_info.loc[i,:]
    X = Particle_Path_Points.loc['MatchedHit_X[0]':'MatchedHit_X[3]'].values
    Y = Particle_Path_Points.loc['MatchedHit_Y[0]':'MatchedHit_Y[3]'].values
    Z = Particle_Path_Points.loc['MatchedHit_Z[0]':'MatchedHit_Z[3]'].values

    data = np.concatenate((X[:, np.newaxis], 
                        Y[:, np.newaxis], 
                        Z[:, np.newaxis]), 
                        axis=1)

    datamean = data.mean(axis=0)

    # uu, dd and vv contain information on the fit. In fact, vv[0] contains the direction of the best fit (least squares)
    uu, dd, vv = np.linalg.svd(data - datamean)

    # Best fit line with length between -2500 and 2500 with 2 datapoints (it's a straight line so that's enough)
    linepts = vv[0] * np.mgrid[-2500:2500:2j][:, np.newaxis]
    LineSlope = vv[0]
    LineSlope1 = np.append(LineSlope1, LineSlope[0])
    LineSlope2 = np.append(LineSlope2, LineSlope[1])
    LineSlope3 = np.append(LineSlope3, LineSlope[2])


    # Shift by the mean to get the line in the right place (centered)
    linepts += datamean

    # Do a little loop where we add the residuals of all four of the points
    for j in range (0,4):
        intersect = intersect_point_line(data[j], linepts[0], linepts[1])
        ResidualsSize += abs(sum(data[j] - intersect[0]))

    PointResiduals = np.append(PointResiduals, ResidualsSize)

    # Singular value decomposition on first line to get its derivative (angle)
    DataFirstTwo = data[0:2,:]
    datamean = DataFirstTwo.mean(axis=0)
    uu, dd, vv = np.linalg.svd(DataFirstTwo - datamean)
    FirstLineAngle = vv[0]
    
    
    # Singular value decomposition on second line to get its derivative (angle)
    DataSecondTwo = data[2:4,:]    
    datamean = DataSecondTwo.mean(axis=0)
    uu, dd, vv = np.linalg.svd(DataSecondTwo - datamean)
    SecondLineAngle = vv[0]
    
    # Finding the angle between the vectors made up by the first and second line. Simply dot product as both lines are normalized (norm=1 for both)
    Angle = np.dot(FirstLineAngle,SecondLineAngle)
    Angles = np.append(Angles, Angle)

    # Singular value decomposition for first three points to get residuals with the fourth point
    DataFirstThree = data[0:3,:]
    datamean = DataFirstThree.mean(axis=0)
    uu, dd, vv = np.linalg.svd(DataFirstThree - datamean)
    
    linepts = vv[0] * np.mgrid[-2500:2500:2j][:, np.newaxis]
    linepts += datamean
    
    intersect = intersect_point_line(data[3], linepts[0], linepts[1])
    ResidualSize = abs(norm(data[3] - intersect[0]))
    
    FourthPointResiduals = np.append(FourthPointResiduals, ResidualSize)
    
    # Singular value decomposition for last three points to get residuals with the first point
    DataLastThree = data[1:4,:]
    datamean = DataLastThree.mean(axis=0)
    uu, dd, vv = np.linalg.svd(DataLastThree - datamean)
    
    linepts = vv[0] * np.mgrid[-2500:2500:2j][:, np.newaxis]
    linepts += datamean
    
    intersect = intersect_point_line(data[0], linepts[0], linepts[1])
    ResidualSize = abs(norm(data[0] - intersect[0]))
    
    FirstPointResiduals = np.append(FirstPointResiduals, ResidualSize)

    if i % 100000 == 0:
        df = test.loc[i-100000:i,:]
        df['PointResiduals'] = pd.Series(PointResiduals)
        df['Angles'] = pd.Series(Angles)
        df['LineSlope1'] = pd.Series(LineSlope1)
        df['LineSlope2'] = pd.Series(LineSlope2)
        df['LineSlope3'] = pd.Series(LineSlope3)
        df['FirstPointResiduals'] = pd.Series(FirstPointResiduals)
        df['FourthPointResiduals'] = pd.Series(FourthPointResiduals)

        with open('./data/testMoreFeatures.csv', 'a') as f:
            df.to_csv(f, header=False)

        countertest += 1

    if i == DataSet.shape[0]-1:
        df = pd.DataFrame([],columns = ["PointResiduals", "Angles", "LineSlope1", "LineSlope2", "LineSlope3", "FirstPointResiduals", "FourthPointResiduals"])
        df['PointResiduals'] = pd.Series(PointResiduals)
        df['Angles'] = pd.Series(Angles)
        df['LineSlope1'] = pd.Series(LineSlope1)
        df['LineSlope2'] = pd.Series(LineSlope2)
        df['LineSlope3'] = pd.Series(LineSlope3)
        df['FirstPointResiduals'] = pd.Series(FirstPointResiduals)
        df['FourthPointResiduals'] = pd.Series(FourthPointResiduals)

        with open('./data/trainMoreFeatures.csv', 'a') as f:
            df.to_csv(f, header=False)

        df = pd.DataFrame([])
        PointResiduals = np.array([])
        Angles = np.array([])
        LineSlope1 = np.array([])
        LineSlope2 = np.array([])
        LineSlope3 = np.array([])
        FourthPointResiduals = np.array([])
        FirstPointResiduals = np.array([])

# Create features for train dataset

print("Finished everything :)")
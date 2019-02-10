import os
from itertools import repeat
import numpy as np
import pandas as pd
from mathutils.geometry import intersect_point_line
from numpy.linalg import norm


# Columns common between both training and test datasets
SIMPLE_FEATURE_COLUMNS = ['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]',
       'avg_cs[1]', 'avg_cs[2]', 'avg_cs[3]', 'ndof', 'MatchedHit_TYPE[0]',
       'MatchedHit_TYPE[1]', 'MatchedHit_TYPE[2]', 'MatchedHit_TYPE[3]',
       'MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]',
       'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
       'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
       'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]',
       'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]',
       'MatchedHit_DX[3]', 'MatchedHit_DY[0]', 'MatchedHit_DY[1]',
       'MatchedHit_DY[2]', 'MatchedHit_DY[3]', 'MatchedHit_DZ[0]',
       'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]',
       'MatchedHit_T[0]', 'MatchedHit_T[1]', 'MatchedHit_T[2]',
       'MatchedHit_T[3]', 'MatchedHit_DT[0]', 'MatchedHit_DT[1]',
       'MatchedHit_DT[2]', 'MatchedHit_DT[3]', 'Lextra_X[0]', 'Lextra_X[1]',
       'Lextra_X[2]', 'Lextra_X[3]', 'Lextra_Y[0]', 'Lextra_Y[1]',
       'Lextra_Y[2]', 'Lextra_Y[3]', 'NShared', 'Mextra_DX2[0]',
       'Mextra_DX2[1]', 'Mextra_DX2[2]', 'Mextra_DX2[3]', 'Mextra_DY2[0]',
       'Mextra_DY2[1]', 'Mextra_DY2[2]', 'Mextra_DY2[3]', 'FOI_hits_N', 'PT', 'P']


# Columns only present in training set
TRAIN_COLUMNS = ["label", "weight"]


FOI_COLUMNS = ["FOI_hits_X", "FOI_hits_Y", "FOI_hits_T",
               "FOI_hits_Z", "FOI_hits_DX", "FOI_hits_DY", "FOI_hits_S"]

# Id column is also present in both training and test datasets
ID_COLUMN = "id"

# Loads the original, huge files
def load_data_csv(path, feature_columns):
    train = pd.concat([
        pd.read_csv(os.path.join(path, "train_part_%i.csv.gz" % i),
                    usecols= [ID_COLUMN] + feature_columns + TRAIN_COLUMNS,
                    index_col=ID_COLUMN)
        for i in (1, 2)], axis=0, ignore_index=True)
    test = pd.read_csv(os.path.join(path, "test_public.csv.gz"),
                       usecols=[ID_COLUMN] + feature_columns, index_col=ID_COLUMN)
    return train, test

# Imports the smaller train and test data files that result from "make_smaller" as csv to pandas dataframe, can also deal with compressed (gzip) files
def load_small_data_csv(path, train_filename, test_filename, feature_columns):
    train = pd.read_csv(os.path.join(path, train_filename)
                    #usecols= [ID_COLUMN] + feature_columns + TRAIN_COLUMNS,
                    #index_col=ID_COLUMN
                    )
    test = pd.read_csv(os.path.join(path, test_filename),
                       usecols=[ID_COLUMN] + feature_columns, index_col=ID_COLUMN
                       )
    try:
        train = train.drop('Unnamed: 0', axis=1)
    except:
        pass
    
    return train, test
    
# Makes dataframes smaller and saves them as csv as the original dataframes can be a bit hard to deal with
def make_smaller(train,test,smaller_by):
       Height = train.shape[0] # Height is the number of columns in the dataframe
       train_small = train[:round((Height/smaller_by))] # Making the train dataframe 100 times smaller
       test_small = test[:round((Height/smaller_by))] # Making the test dataframe 100 times smaller

       # Saving the smaller test and train sets
       train_small.to_csv(r"./data" + r"/train_smaller" + str(smaller_by) + ".csv.gz", compression = 'gzip')
       test_small.to_csv(r"./data" + r"/test_smaller" + str(smaller_by) + ".csv.gz", compression = 'gzip')

# Returns array of summed residuals per particle (i-th entry in array correlates to i-th particle in dataframe). Currently only does for train.
def kink(DataSet):
    Location_info = DataSet.loc[: , "MatchedHit_X[0]":"MatchedHit_Z[3]"]

    PointResiduals = np.array([])
    Angles = np.array([])
    LineSlope = np.array([])
    FourthPointResiduals = np.array([])
    FirstPointResiduals = np.array([])

    for i in range(0,DataSet.shape[0]):
    
        ResidualsSize = 0
    
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
        LineSlope = np.append(LineSlope, vv[0])

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
    
    return PointResiduals, Angles, LineSlope, FirstPointResiduals, FourthPointResiduals
import tensorflow as tf
import keras
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
import utils
import scoring
import numpy as np
import json

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization

DATA_PATH = "./data/"

np.random.seed(0)
### Importing and structuring data ###

#train, test = utils.load_data( bDATA_PATH)
print("Importing data...")
train, test = utils.load_data_csv(DATA_PATH, utils.SIMPLE_FEATURE_COLUMNS)

train_part, val_part = train_test_split(train, test_size=0.20, shuffle=True)

x_train = train_part.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values
x_val   =  val_part.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values
y_train = train_part.loc[:, ["label"]].values
y_val = val_part.loc[:, ["label"]].values

# turn labels into categorical classes
classes = [0,1]
y_train = keras.utils.to_categorical(y_train, num_classes=len(classes))
y_val   = keras.utils.to_categorical(y_val,   num_classes=len(classes))

### Defining utils ###

# Rectified linear unit
def relu(x):
  return np.array([ (i>0) * abs(i) for i in x ])

#### looping over each of the best networks to get some significant results ####
print("Running network")
for i in range (0,1):
    
    INIT_LEARNINGRATE = 0.00958
    BATCH_SIZE = 16  # should be a factor of len(x_train) and len(x_val) etc.
    EPOCHS = 3

    assert len(y_train) == len(x_train), "x_train and y_train not same length!"
    #assert len(y_train) % BATCH_SIZE == 0, "batch size should be multiple of training size,{0}/{1}".format(len(y_train),BATCH_SIZE)

    K.clear_session()

    model = Sequential()
    model.add(Dense(165, activation='relu', input_shape=( len( utils.SIMPLE_FEATURE_COLUMNS ), ))) #length = input vars
    model.add(BatchNormalization())
    model.add(Dense(160 , activation='relu'))
    model.add(Dense(194)) # muon and 'other'
    model.add(Dropout(0.00453))
    model.add(Dense(16 )) # muon and 'other'
    model.add(Dense( len(classes) ))
    model.add(Activation("softmax")) # output probabilities

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.adamax(lr=INIT_LEARNINGRATE),
        metrics=['accuracy'] 
        )


    model.fit(
        x_train, y_train,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        validation_data = (x_val, y_val),
        shuffle = True
        )

    #model.save_model("keras_basic_model.xgb")


    # score

    validation_predictions = model.predict_proba(val_part.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values)[:, 1]
    result = scoring.rejection90(val_part.label.values, validation_predictions, sample_weight=val_part.weight.values)
    FirstNet = np.append(FirstNet, result)

    predictions = model.predict_proba(test.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values)[:, 1]

pd.DataFrame(data={"prediction": predictions}, index=test.index).to_csv(
    "keras_basic_submission.csv", index_label=utils.ID_COLUMN)
#%%
import utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#%%
path = r"./data/"


#%%
train, test = utils.load_data_csv(path, utils.SIMPLE_FEATURE_COLUMNS)


#%%
df_y = train["label"] # df_y consists of all the labels, target variable


#%%
df_x = train[utils.SIMPLE_FEATURE_COLUMNS] # df_x consists of all of the other data

#%%
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25)


#%%
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(x_train, y_train)


#%%
predictions = rf.predict_proba(test.loc[:, utils.SIMPLE_FEATURE_COLUMNS].values)[:, 1]

pd.DataFrame(data={"prediction": predictions}, index=test.index).to_csv(
    "randomforest_submission.csv", index_label=utils.ID_COLUMN)









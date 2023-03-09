from uuid import uuid4

import numpy as np
import pandas as pd

from surrogate.acquisition import ExpectedImprovementRanking
from surrogate.dense import DenseRandomForestRegressor
from surrogate.data import Hdf5Dataset


# ----------------------------------------------------
run_code = F'RandomForest_ei_{uuid4().hex[::4]}'
y_ref = Hdf5Dataset('E7_07_XeKr_values.hdf5')

model = DenseRandomForestRegressor(data_set=Hdf5Dataset('E7_05.hdf5'))
acquisitor = ExpectedImprovementRanking()

X_train_ind = list(np.random.choice(len(y_ref), 50))
y_train = y_ref[X_train_ind]


for itr in range(50, 370):        
    
    model.fit(X_train_ind, y_train)
    mu, std = model.predict()
    
    alpha = acquisitor.score_points(mu, std, y_train.max())
    alpha_ranked = np.argsort(alpha)[::-1]
    to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
    
    X_train_ind.append(to_sample)    
    y_train = y_ref[X_train_ind]
    
    df = pd.DataFrame({'x': X_train_ind, 'y': y_train.ravel()})
    df.to_csv(F'ami_{run_code}.csv', index=False)
    
    
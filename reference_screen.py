from uuid import uuid4

import numpy as np
import pandas as pd

from surrogate.acquisition import GreedyNRanking
from surrogate.dense import DenseGaussianProcessregressor
from surrogate.data import Hdf5Dataset


# ----------------------------------------------------
run_code = uuid4().hex[::4]
X_ref = Hdf5Dataset('E7_05.hdf5')
y_ref = Hdf5Dataset('E7_07_XeKr_values.hdf5')

model = DenseGaussianProcessregressor(data_set=X_ref)
acquisitor = GreedyNRanking(n_opt=100)

X_train_ind = list(np.random.choice(len(X_ref), 1))
y_train = y_ref[X_train_ind]


for itr in range(1, 370):        
    
    print(X_train_ind)
    model.fit(X_train_ind, y_train)
    posterior = model.sample_y(n_samples=100)
    
    alpha = abs(acquisitor.score_points(posterior))
    alpha_ranked = np.argsort(alpha)[::-1]
    to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
    
    X_train_ind.append(to_sample)    
    y_train = abs(y_ref[X_train_ind])
    
    df = pd.DataFrame({'x': X_train_ind, 'y_abs': y_train.ravel()})
    df.to_csv(F'ami_ref_run_{run_code}.csv', index=False)
    
    
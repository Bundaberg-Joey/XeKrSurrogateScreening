from uuid import uuid4
import argparse

import numpy as np
import pandas as pd

from surrogate.acquisition import GreedyNRanking
from surrogate.dense import DenseTanimotoGPR, DenseMaternGPR, DenseGaussianProcessregressor, EnsembleGPR
from surrogate.data import Hdf5Dataset


# ----------------------------------------------------
run_code = F'Ensemble_{uuid4().hex[::4]}'
y_ref = Hdf5Dataset('E7_07_XeKr_values.hdf5')

rbf_model = DenseGaussianProcessregressor(data_set=Hdf5Dataset('E7_05.hdf5'))
tanimoto_model = DenseTanimotoGPR(data_set=Hdf5Dataset('E7_11_PCFP_1_ind_1024.hdf5'))
model = EnsembleGPR(rbf_model, tanimoto_model)

acquisitor = GreedyNRanking(n_opt=100)

X_train_ind = list(np.random.choice(len(y_ref), 1))
y_train = y_ref[X_train_ind]


for itr in range(1, 370):        
    
    print(X_train_ind)
    model.fit(X_train_ind, y_train)
    posterior = model.sample_y(n_samples=50)  # not 100 since will sample 50 from each model in ensemble --> 100 total
    
    alpha = acquisitor.score_points(posterior)
    alpha_ranked = np.argsort(alpha)[::-1]
    to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
    
    X_train_ind.append(to_sample)    
    y_train = y_ref[X_train_ind]
    
    df = pd.DataFrame({'x': X_train_ind, 'y': y_train.ravel()})
    df.to_csv(F'ami_ensemble_run_{run_code}.csv', index=False)
    
    
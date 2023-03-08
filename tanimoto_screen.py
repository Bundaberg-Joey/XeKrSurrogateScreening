from uuid import uuid4
import argparse

import numpy as np
import pandas as pd

from surrogate.acquisition import GreedyNRanking
from surrogate.dense import DenseTanimotoGPR
from surrogate.data import Hdf5Dataset


# ----------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help='fp file to use')
args = parser.parse_args()

fp_file = str(args.f)
assert fp_file in [F'E7_11_PCFP_{i}_ind_1024.hdf5' for i in range(3)]


# ----------------------------------------------------
run_code = F'Tanimoto_{fp_file}_{uuid4().hex[::4]}'
X_ref = Hdf5Dataset(fp_file)
y_ref = Hdf5Dataset('E7_07_XeKr_values.hdf5')

model = DenseTanimotoGPR(data_set=X_ref)
acquisitor = GreedyNRanking(n_opt=100)

X_train_ind = list(np.random.choice(len(X_ref), 1))
y_train = y_ref[X_train_ind]


for itr in range(1, 370):        
    
    print(X_train_ind)
    model.fit(X_train_ind, y_train)
    posterior = model.sample_y(n_samples=100)
    
    alpha = acquisitor.score_points(posterior)
    alpha_ranked = np.argsort(alpha)[::-1]
    to_sample = [i for i in alpha_ranked if i not in X_train_ind][0]
    
    X_train_ind.append(to_sample)    
    y_train = y_ref[X_train_ind]
    
    df = pd.DataFrame({'x': X_train_ind, 'y': y_train.ravel()})
    df.to_csv(F'ami_ref_run_{run_code}.csv', index=False)
    
    
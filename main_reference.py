import argparse
from uuid import uuid4

import pandas as pd
import numpy as np

from surrogate.acquisition import GreedyNRanking, EiRanking
from surrogate.dense import DenseGaussianProcessregressor, DenseRandomForestRegressor
from surrogate.data import Hdf5Dataset

from ranking_models import PosteriorRanker, ExpectedImprovementRanker


# ----------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-r', type=str, help='Ranker to use')
args = parser.parse_args()

ranker_choice = args.r
run_code = F'{ranker_choice}_{uuid4().hex[::4]}'

# ----------------------------------------------------

X_ref = Hdf5Dataset('Ex7_05_descriptors.hdf5')
y_ref = Hdf5Dataset('Ex7_05_selectivity.hdf5')[:].ravel()

greedy_n_ranker = PosteriorRanker(
    model=DenseGaussianProcessregressor(data_set=X_ref),
    acquisitor=GreedyNRanking(n_opt=100),
    n_post=100
    )

ei_ranker = ExpectedImprovementRanker(
    model=DenseRandomForestRegressor(data_set=X_ref),
    acquisitor=EiRanking()
)

ranker = {'ei': ei_ranker, 'greedy': greedy_n_ranker}[ranker_choice]


X_train_ind = np.random.choice(len(y_ref.ravel()), size=1, replace=False)
#X_train_ind = list(pd.read_csv('Ex7_05_indices_sampled.txt', header=None)[0])
y_train = y_ref[X_train_ind]


for itr in range(370):        
    
    ranker.fit(X_train_ind, y_train)
    
    alpha = ranker.determine_alpha()
    rankings = np.argsort(alpha)[::-1]    
    to_sample = [i for i in rankings if i not in X_train_ind][0]
        
    X_train_ind.append(to_sample)    
    y_train = abs(y_ref[X_train_ind])
    
    df = pd.DataFrame({'x': X_train_ind, 'y': y_train.ravel()})
    df.to_csv(F'ami_ref_run_{run_code}.csv', index=False)
    
    
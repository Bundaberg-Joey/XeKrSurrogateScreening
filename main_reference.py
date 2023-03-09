import argparse

import pandas as pd

from surrogate.acquisition import GreedyNRanking, EiRanking
from surrogate.dense import DenseGaussianProcessregressor, DenseRandomForestRegressor
from surrogate.data import Hdf5Dataset

from ranking_models import PosteriorRanker, ExpectedImprovementRanker


# ----------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-r', type=str, help='Ranker to use')
args = parser.parse_args()

ranker_choice = args.r

# ----------------------------------------------------

X_ref = Hdf5Dataset('Ex7_01_physical.hdf5')
y_ref = Hdf5Dataset('Ex7_02_selectivity.hdf5')[:].ravel()

greedy_n_ranker = PosteriorRanker(
    model=DenseGaussianProcessregressor(data_set=X_ref),
    acquisitor=GreedyNRanking(n_opt=100),
    n_post=100
    )

ei_ranker = ExpectedImprovementRanker(
    model=DenseRandomForestRegressor(dataset=X_ref),
    acquisitor=EiRanking()
)

ranker = {'ei': ei_ranker, 'greedy': greedy_n_ranker}[ranker_choice]


X_train_ind = list(pd.read_csv('Ex7_03_sample_indices.txt', header=None)[0])
y_train = y_ref[X_train_ind]


for itr in range(370):        
    
    ranker.fit(X_train_ind, y_train)
    ranked_remaining = ranker.rank([ix for ix in range(len(X_ref)) if ix not in X_train_ind])
    to_sample = ranked_remaining[0]
        
    X_train_ind.append(to_sample)    
    y_train = abs(y_ref[X_train_ind])
    
    df = pd.DataFrame({'x': X_train_ind, 'y': y_train.ravel()})
    df.to_csv(F'ami_ref_run_{ranker_choice}.csv', index=False)
    
    
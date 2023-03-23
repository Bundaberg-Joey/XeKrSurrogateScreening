import argparse
from uuid import uuid4

import pandas as pd
import numpy as np

from surrogate.acquisition import EiRanking
from surrogate.dense import DenseGaussianProcessregressor, DenseRandomForestRegressor
from surrogate.data import Hdf5Dataset

from ranking_models import ExpectedImprovementRanker

# ----------------------------------------------------

class RandomRanker:

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.n = len(self.dataset)

    def fit(self, X, y):
        pass

    def determine_alpha(self):
        return np.random.uniform(size=self.n)


# ----------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, help='model to use')
args = parser.parse_args()

ranker_choice = args.m
run_code = F'{ranker_choice}_{uuid4().hex[::4]}'

# ----------------------------------------------------

X_ref = Hdf5Dataset('Ex7_05_descriptors.hdf5')
y_ref = Hdf5Dataset('Ex7_05_selectivity.hdf5')[:].ravel()

rf_ranker = ExpectedImprovementRanker(
    model=DenseRandomForestRegressor(data_set=X_ref),
    acquisitor=EiRanking()
)

gp_ranker = ExpectedImprovementRanker(
    model=DenseGaussianProcessregressor(data_set=X_ref),
    acquisitor=EiRanking()
)

random_ranker = RandomRanker(dataset=X_ref)

ranker = {'rf': rf_ranker, 'gp': gp_ranker, 'random': random_ranker}[ranker_choice]

X_train_ind = list(pd.read_csv('Ex7_05_indices_sampled.txt', header=None)[0])
y_train = y_ref[X_train_ind]


for itr in range(341):        
    
    ranker.fit(X_train_ind, y_train)
    
    alpha = ranker.determine_alpha()
    rankings = np.argsort(alpha)[::-1]    
    to_sample = [i for i in rankings if i not in X_train_ind][0]
        
    X_train_ind.append(to_sample)    
    y_train = abs(y_ref[X_train_ind])
    
    df = pd.DataFrame({'x': X_train_ind, 'y': y_train.ravel()})
    df.to_csv(F'ami_ref_run_{run_code}.csv', index=False)
    
    

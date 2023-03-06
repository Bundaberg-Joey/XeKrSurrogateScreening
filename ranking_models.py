from typing import Sequence, Iterator, Union

import numpy as np
from numpy.typing import NDArray

import ami.abc
from ami.abc import SchemaInterface, RankerInterface, Feature, Target
from ami.abc.ranker import Index
from ami.schema import Schema

from surrogate.dense import DenseGaussianProcessregressor
from surrogate.acquisition import GreedyNRanking, ThompsonRanking


# ---------------------------------------------------------------------------------------


class RandomRanker(RankerInterface):
    """Used to randomly sample at start of screening as initial ranker.
    Could instead use clustering algorithm / other approach to select initial indices for screening.
    """    
    def rank(self, x: Sequence[Feature]) -> Iterator[Index]:
        my_rank = np.arange(len(x), dtype=int)
        np.random.shuffle(my_rank)
        return my_rank

    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        pass

    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('index', int)],
            output_schema=[('target', float)]
        )


# ---------------------------------------------------------------------------------------

class SurrogateModelRanker(ami.abc.RankerInterface):
    
    def __init__(self, model, acquisitor) -> None:
        self.model = model
        self.acquisitor = acquisitor
        
    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        """Fit model to passed data points

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            indices of data points to use.
            
        y_val : NDArray[np.int_]
            taret values of data points.
        """
        self.model.fit(x, y)        
        
    def rank(self, x: Sequence[Feature]) -> Iterator[Index]:
        """Rank the passed indices from highest to lowest.
        Highest ranked are highest recommended to be sampled.

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            _description_

        Returns
        -------
        NDArray[np.float_]
            ranked highest to lowest, element 0 is largest ranked, element -1 is lowest ranked.
        """
        rankings = self.rank_points(x)
        return rankings  # index of largest alpha is first
    
    def schema(self) -> SchemaInterface:
        return Schema(
            input_schema=[('index', int)],
            output_schema=[('target', float)]
        )


class PosteriorSurrogateRanker(SurrogateModelRanker):
    def __init__(self, 
                 model: DenseGaussianProcessregressor, 
                 acquisitor: Union[GreedyNRanking, ThompsonRanking], 
                 n_post: int=50, 
                 take_absolute=True
                 ) -> None:
        """
        Parameters
        ----------
        model : model object
            Must have `fit(X_ind, y)` and `sample_y(n_samples)` methods.            
        """
        self.model = model
        self.acquisitor = acquisitor
        self.n_post = int(n_post)
        self.take_absolute = bool(take_absolute)
        
    def rank_points(self, X_ind: NDArray[np.int_]) -> NDArray[np.int_]:
        """Determine the alpha (ranking values) for each data point in `X_ind`.
        Performs self.n_post posterior samples for each instance in self.model and determines the absolute values.

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            indices of data points to use.

        Returns
        -------
        NDArray[np.int_]
            ranked alpha terms for the specified indices ranked highest t lowest where highest is most recommended.
        """
        posterior = self.model.sample_y(n_samples=self.n_post)
        
        if self.take_absolute:
            posterior = abs(posterior)
        
        alpha = self.acquisitor.score_points(posterior)
        alpha = alpha[X_ind]
        ranked_points = np.argsort(alpha)[::-1]
        return ranked_points

# ---------------------------------------------------------------------------------------
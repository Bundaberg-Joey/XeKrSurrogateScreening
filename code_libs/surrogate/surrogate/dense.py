import numpy as np
from numpy.typing import NDArray
import gpflow

from surrogate.data import Hdf5Dataset
from surrogate.kernel import SafeMatern12, SafeMatern32, SafeMatern52, TanimotoKernel


# ------------------------------------------------------------------------------------------------------------------------------------


class DenseGaussianProcessregressor:
    
    def __init__(self, data_set: Hdf5Dataset) -> None:
        self.data_set = data_set
        self.model = None
        self._model_built = False
        
    def build_model(self, X: NDArray[NDArray[np.float_]], y: NDArray[np.float_]) -> gpflow.models.GPR:
        """Initialise and return the gpflow model (will be optimised when `fit` is called).
        Just a covenience method for subclassing to create different dense models more flexibly.

        Parameters
        ----------
        X : NDArray[NDArray[np.float_]]
            Feature matrix to fit model to, rows are entries and columns are features.
            
        y : NDArray[np.float_]
            Target values for passed entries.

        Returns
        -------
        gpflow.models.GPR
        """
        if y.ndim != 2:
            y = y.reshape(-1, 1)  # gpflow needs column vector for target
        
        model = gpflow.models.GPR(
        data=(X, y), 
        kernel=gpflow.kernels.RBF(lengthscales=np.ones(X.shape[1])),
        mean_function=gpflow.mean_functions.Constant()
        )
        return model
        
    def fit(self, X_ind: NDArray[np.int_], y_val: NDArray[np.float_]) -> None:
        """Fit the backend gpflow model to the passed data.

        Parameters
        ----------
        X_ind : NDArray[np.int_]
            Indices of data points to use when fitting.
            They are also incorporated into the inducing feature matrix each time fit is called.
            
        y_val : NDArray[np.float_]
            Target values for each entry. 
            
        Returns
        -------
        None
        """
        X = self.data_set[X_ind]
        self.model = self.build_model(X, y_val)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.model.training_loss, self.model.trainable_variables)  
        self._model_built = True

    def sample_y(self, n_samples=1):
        if self._model_built:
            posterior = self.model.predict_f_samples(self.data_set[:], num_samples=int(n_samples))
            return posterior.numpy().T[0]
        else:
            raise ValueError('Model not yet fit to data.')


# ------------------------------------------------------------------------------------------------------------------------------------


class DenseMaternGPR(DenseGaussianProcessregressor):
    
    def __init__(self, data_set: Hdf5Dataset, matern) -> None:
        super().__init__(data_set)
        self.base_kernel = {12: SafeMatern12, 32: SafeMatern32, 52: SafeMatern52}[matern]
    
    def build_model(self, X: NDArray[NDArray[np.float_]], y: NDArray[np.float_]) -> gpflow.models.GPR:
        if y.ndim != 2:
            y = y.reshape(-1, 1)  # gpflow needs column vector for target
        
        model = gpflow.models.GPR(
        data=(X, y), 
        kernel=self.base_kernel(lengthscales=np.ones(X.shape[1])),
        mean_function=gpflow.mean_functions.Constant()
        )
        return model


class DenseTanimotoGPR(DenseGaussianProcessregressor):
    def build_model(self, X: NDArray[NDArray[np.float_]], y: NDArray[np.float_]) -> gpflow.models.GPR:
        if y.ndim != 2:
            y = y.reshape(-1, 1)  # gpflow needs column vector for target
        
        model = gpflow.models.GPR(
        data=(X, y), 
        kernel=TanimotoKernel(),
        mean_function=gpflow.mean_functions.Constant()
        )
        return model


# ------------------------------------------------------------------------------------------------------------------------------------


class EnsembleGPR:
    
    def __init__(self, ma: DenseGaussianProcessregressor, mb: DenseGaussianProcessregressor) -> None:
        self.ma = ma
        self.mb = mb
        
    def fit(self, X_ind: NDArray[np.int_], y_val: NDArray[np.float_]) -> None:
        self.ma.fit(X_ind, y_val)
        self.mb.fit(X_ind, y_val)
            
    def sample_y(self, n_samples=1):
        samples = [m.sample_y(n_samples=n_samples) for m in [self.ma, self.mb]]
        return np.hstack(samples)  # n_entries_in_X, n_samples * len(self.models)
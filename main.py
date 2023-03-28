import argparse
from uuid import uuid4

import pandas as pd

from ami.mp.configuration import Configuration
from ami.data_manager import InMemoryDataManager
from ami.scheduler import SerialSchedulerFactory
from ami.worker import ShareMemorySingleThreadWorkerFactory
from ami.worker_pool import SingleNodeWorkerPoolFactory
from ami.option import Some

from surrogate.acquisition import EiRanking
from surrogate.dense import DenseGaussianProcessregressor, DenseRandomForestRegressor
from surrogate.data import Hdf5Dataset

from ranking_models import ExpectedImprovementRanker, RandomRanker
from raspa import XeKrSeparation


# ---------------------------------------------------------------------------------------
# collect args

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, help='Total number of MOFs to screen.', default=344)
parser.add_argument('-r', type=str, help='Ranker to use')
args = parser.parse_args()

code = uuid4().hex[::4]
pool_size = 1
n_tasks = args.n
ranker_choice = args.r
run_code = F'{ranker_choice}_{code}'

# ---------------------------------------------------------------------------------------
# set up ML code
hdf5_dataset = Hdf5Dataset('Ex7_05_descriptors_2.hdf5')
prior_values = pd.read_csv('Ex7_05_init_random_sample_2.csv')
X_init, y_init = prior_values['index'].tolist(), prior_values['selectivity'].tolist()

model = DenseGaussianProcessregressor(data_set=hdf5_dataset)

gp_ranker = ExpectedImprovementRanker(
    model=DenseGaussianProcessregressor(data_set=hdf5_dataset),
    acquisitor=EiRanking()
    )

rf_ranker = ExpectedImprovementRanker(
    model=DenseRandomForestRegressor(dataset=hdf5_dataset),
    acquisitor=EiRanking()
)

surrogate_ranker = {'gp': gp_ranker, 'rf': rf_ranker}[ranker_choice]

# # ---------------------------------------------------------------------------------------
# Set up AMI code
calc = XeKrSeparation.from_template_folder(F"internal_workdir_{run_code}", "raspa_template")
init_ranker = RandomRanker()
pool = SingleNodeWorkerPoolFactory()
pool.set("ncpus", pool_size)

config = Configuration(
    scheduler=SerialSchedulerFactory(),
    worker=ShareMemorySingleThreadWorkerFactory(),
    data=InMemoryDataManager.from_indexed_list_in_file("Ex7_05_cif_list_2.txt",
                                                       calc_schema=calc.schema(),
                                                       surrogate_schema=surrogate_ranker.schema(),
                                                       csv_filename=F'ami_output_{run_code}.txt'
                                                       ),
    truth=calc,
    pool=pool,
    initial_ranker=init_ranker,
    ranker=surrogate_ranker,
)

# incorporating data from initial sample for consistency
for x_, y_ in zip(X_init, y_init):
    config.data.set_result(x_, Some(y_))

# # ---------------------------------------------------------------------------------------
# Run screening
runner = config.build()
runner.run(n_tasks)


# # ---------------------------------------------------------------------------------------

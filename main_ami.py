import argparse
from uuid import uuid4

from ami.mp.configuration import Configuration
from ami.data_manager import InMemoryDataManager
from ami.scheduler import SerialSchedulerFactory
from ami.worker import ShareMemorySingleThreadWorkerFactory
from ami.worker_pool import SingleNodeWorkerPoolFactory

from surrogate.acquisition import GreedyNRanking, EiRanking
from surrogate.dense import DenseGaussianProcessregressor, DenseRandomForestRegressor
from surrogate.data import Hdf5Dataset

from ranking_models import PosteriorRanker, ExpectedImprovementRanker, RandomRanker
from raspa import XeKrSeparation


# ---------------------------------------------------------------------------------------
# collect args

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, help='Total number of MOFs to screen.', default=370)
parser.add_argument('-r', type=str, help='Ranker to use')
args = parser.parse_args()

code = uuid4().hex[::4]
pool_size = 1
n_tasks = args.n
ranker_choice = args.r
run_code = F'{ranker_choice}_{code}'

# ---------------------------------------------------------------------------------------
# set up ML code
hdf5_dataset = Hdf5Dataset('Ex7_01_physical.hdf5')
model = DenseGaussianProcessregressor(data_set=hdf5_dataset)

greedy_n_ranker = PosteriorRanker(
    model=DenseGaussianProcessregressor(data_set=hdf5_dataset),
    acquisitor=GreedyNRanking(n_opt=100),
    n_post=100
    )

ei_ranker = ExpectedImprovementRanker(
    model=DenseRandomForestRegressor(dataset=hdf5_dataset),
    acquisitor=EiRanking()
)

surrogate_ranker = {'ei': ei_ranker, 'greedy': greedy_n_ranker}[ranker_choice]

cached_results = Hdf5Dataset('E7_07_XeKr_values.hdf5')

# # ---------------------------------------------------------------------------------------
# Set up AMI code
calc = XeKrSeparation.from_template_folder(F"internal_workdir_{run_code}", "raspa_template")
init_ranker = RandomRanker()
pool = SingleNodeWorkerPoolFactory()
pool.set("ncpus", pool_size)

config = Configuration(
    scheduler=SerialSchedulerFactory(),
    worker=ShareMemorySingleThreadWorkerFactory(),
    data=InMemoryDataManager.from_indexed_list_in_file("E7_05_cif_list.txt",
                                                       calc_schema=calc.schema(),
                                                       surrogate_schema=surrogate_ranker.schema(),
                                                       csv_filename=F'ami_output_{run_code}.txt'
                                                       ),
    truth=calc,
    pool=pool,
    initial_ranker=init_ranker,
    ranker=surrogate_ranker,
)


# # ---------------------------------------------------------------------------------------
# Run screening
runner = config.build()
runner.run(n_tasks)


# # ---------------------------------------------------------------------------------------

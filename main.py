import argparse

from ami.mp.configuration import Configuration
from ami.data_manager import InMemoryDataManager
from ami.scheduler import SerialSchedulerFactory
from ami.worker import ShareMemorySingleThreadWorkerFactory
from ami.worker_pool import SingleNodeWorkerPoolFactory

from surrogate.acquisition import GreedyNRanking
from surrogate.dense import DenseGaussianProcessregressor
from surrogate.data import Hdf5Dataset

from ranking_models import GreedySeparationRanker, RandomRanker
from raspa import XeKrSeparation


# ---------------------------------------------------------------------------------------
# collect args

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help='Filename for AMI results.', default='ami_output.txt')
parser.add_argument('-p', type=int, help='Number of CPUs to run in pool.', default=1)
parser.add_argument('-n', type=int, help='Total number of MOFs to screen.', default=72)
args = parser.parse_args()

ami_filename = args.f
pool_size = args.p
n_tasks = args.n


# ---------------------------------------------------------------------------------------
# set up ML code
hdf5_dataset = Hdf5Dataset('E7_05.hdf5')

model = DenseGaussianProcessregressor(data_set=hdf5_dataset)

surrogate_ranker = GreedySeparationRanker(
    model=model, 
    acquisitor=GreedyNRanking(n_opt=100),
    n_post=100
)


# # ---------------------------------------------------------------------------------------
# Set up AMI code
calc = XeKrSeparation.from_template_folder("internal_workdir", "raspa_template")
init_ranker = RandomRanker()
pool = SingleNodeWorkerPoolFactory()
pool.set("ncpus", pool_size)

config = Configuration(
    scheduler=SerialSchedulerFactory(),
    worker=ShareMemorySingleThreadWorkerFactory(),
    data=InMemoryDataManager.from_indexed_list_in_file("E7_05_cif_list.txt",
                                                       calc_schema=calc.schema(),
                                                       surrogate_schema=surrogate_ranker.schema(),
                                                       csv_filename=ami_filename
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

import argparse
from uuid import uuid4

from ami.mp.configuration import Configuration
from ami.data_manager import InMemoryDataManager
from ami.scheduler import SerialSchedulerFactory
from ami.worker import ShareMemorySingleThreadWorkerFactory
from ami.worker_pool import SingleNodeWorkerPoolFactory

from surrogate.acquisition import GreedyNRanking, ThompsonRanking
from surrogate.dense import DenseGaussianProcessregressor
from surrogate.data import Hdf5Dataset

from ranking_models import PosteriorSurrogateRanker, RandomRanker
from raspa import XeKrSeparation, CachedXeKrseparation


# ---------------------------------------------------------------------------------------
# collect args

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=int, help='Number of CPUs to run in pool.', default=1)
parser.add_argument('-n', type=int, help='Total number of MOFs to screen.', default=3)
parser.add_argument('-r', type=str, help='Ranking', default='T')
parser.add_argument('-a', type=str, help='use absolute posterior or not', default='False')
args = parser.parse_args()

code = uuid4().hex[::4]
pool_size = args.p
n_tasks = args.n
ranking = {'G': GreedyNRanking(), 'T': ThompsonRanking()}[args.r]
use_abs = {'True': True, 'False': False }[args.a]


run_code = F'{n_tasks}_{args.r}_{use_abs}_{code}'

# ---------------------------------------------------------------------------------------
# set up ML code
hdf5_dataset = Hdf5Dataset('E7_05.hdf5')

model = DenseGaussianProcessregressor(data_set=hdf5_dataset)

surrogate_ranker = PosteriorSurrogateRanker(
    model=model, 
    acquisitor=ranking,
    n_post=1 if args.r == 'T' else 100,
    take_absolute=use_abs
)

print(surrogate_ranker.n_post)

cached_results = Hdf5Dataset('E7_07_XeKr_values.hdf5')

# # ---------------------------------------------------------------------------------------
# Set up AMI code
calc = CachedXeKrseparation(dataset=cached_results)
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

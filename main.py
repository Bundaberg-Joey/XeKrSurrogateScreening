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
parser.add_argument('-n', type=int, help='Total number of MOFs to screen.', default=370)
parser.add_argument('-p', type=int, help='posterior samples', default=100)
parser.add_argument('-o', type=int, help='greedy variant', default=100)
args = parser.parse_args()

code = uuid4().hex[::4]
pool_size = 1
n_tasks = args.n
n_opt = int(args.o)
n_post = int(args.p)


run_code = F'GreedyN_{n_opt}_{n_post}_{code}'

# ---------------------------------------------------------------------------------------
# set up ML code
hdf5_dataset = Hdf5Dataset('E7_05.hdf5')

model = DenseGaussianProcessregressor(data_set=hdf5_dataset)

surrogate_ranker = PosteriorSurrogateRanker(
    model=model, 
    acquisitor=GreedyNRanking(n_opt=n_opt),
    n_post=n_post,
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

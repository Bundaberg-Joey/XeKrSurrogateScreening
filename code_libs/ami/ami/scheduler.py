from dataclasses import dataclass, field
from typing import Tuple, Sequence, Optional

import numpy as np

import ami.abc
from ami.abc.scheduler import Index
from ami.factory import DataclassFactory
from ami.option import Nothing, Some
from ami.option import Option
from ami.serialized_opaque import SerializedOpaque
from ami.surrogate_input import SurrogateInput


@dataclass(slots=True)
class InternalState:
    ptr: int = 0
    dirty_count: int = 0
    threshold: int = 0
    ranked_unknown_indices: Sequence[Index] = ()

    def next(self) -> Index:
        assert len(self.ranked_unknown_indices) > self.ptr
        idx = self.ranked_unknown_indices[self.ptr]
        self.ptr += 1
        return idx

    def reset(self, ranks: Sequence[Index]):
        self.dirty_count = 0
        self.ptr = 0
        self.ranked_unknown_indices = ranks

    def is_dirty(self):
        return self.dirty_count > self.threshold

    def set_dirty(self):
        self.dirty_count += 1

    def set_threshold(self, threshold: int):
        assert threshold >= 0
        self.threshold = threshold


@dataclass(slots=True, frozen=True)
class SerialScheduler(ami.abc.SchedulerInterface):
    data_manager: ami.abc.DataManagerInterface
    worker_pool: ami.abc.WorkerPoolInterface
    initial_ranker: ami.abc.RankerInterface
    surrogate_schema: ami.abc.SchemaProviderInterface
    truth_schema: ami.abc.SchemaProviderInterface
    threshold: int = 0
    _state: InternalState = field(init=False, default_factory=InternalState)

    def __post_init__(self):
        idx, ranker_input = self.ranker_inputs()
        self.initial_ranker.fit(ranker_input.known_x, ranker_input.known_y)
        local_rank = self.initial_ranker.rank(ranker_input.unknown_x)
        glob_rank = np.asarray(idx)[np.asarray(local_rank)]
        self.set_ranks(glob_rank)
        self._state.set_threshold(self.threshold)

    def set_result(self, index: Index, value: Option[SerializedOpaque]):
        self.data_manager.set_result(index, value)
        self._state.set_dirty()

    def set_ranks(self, ranks: Optional[Sequence[Index]]):
        if ranks is None:
            return
        self._state.reset(ranks)

    def needs_new_ranking(self) -> bool:
        return self._state.is_dirty()

    def ranker_inputs(self) -> Tuple[Sequence[Index], SurrogateInput]:
        indices = self.data_manager.available_for_calculation()
        unknown_x = self.data_manager.unknown()
        known_x, known_y = self.data_manager.known()
        assert len(unknown_x) == len(indices)
        return indices, SurrogateInput(known_x, known_y, unknown_x)

    def next(self) -> Index:
        return self._state.next()

    def parameters(self, index: Index) -> SerializedOpaque:
        return self.data_manager.parameters(index).unwrap()


@dataclass(slots=True, frozen=True)
class SerialSchedulerFactory(DataclassFactory, ami.abc.SchedulerFactoryInterface):
    """Defines a 'Scheduler' factory/builder"""

    dataclass = SerialScheduler

    def set_data_manager(self, data_manager: ami.abc.data_manager.DataManagerInterface):
        self.set("data_manager", data_manager)

    def set_ranker_schema(self, schema: ami.abc.schema.SchemaProviderInterface) -> None:
        self.set("surrogate_schema", schema)

    def set_truth_schema(self, schema: ami.abc.schema.SchemaProviderInterface) -> None:
        self.set("truth_schema", schema)

    def set_worker_pool(self, worker_pool: ami.abc.worker_pool.WorkerPoolInterface) -> None:
        self.set("worker_pool", worker_pool)

    def set_initial_ranker(self, ranker: ami.abc.ranker.RankerInterface) -> None:
        self.set("initial_ranker", ranker)

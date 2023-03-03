from dataclasses import dataclass, MISSING
from io import IOBase
from pathlib import Path
from typing import Collection, Sequence, Tuple, List, Union

import numpy as np

import ami.abc
from ami.abc import Target, Feature
from ami.abc.calculator import OpaqueParameters
from ami.option import Option, Nothing, Some
from ami.result import Result, Ok
from ami.schema import Schema

Index = int


@dataclass(slots=True, frozen=True)
class InMemoryStateMachine(ami.abc.StateMachineInterface):
    available: np.ndarray[bool] = MISSING
    done: np.ndarray[bool] = MISSING
    failed: np.ndarray[bool] = MISSING

    @classmethod
    def from_size(cls, size: int):
        return cls(
            available=np.ones(size, dtype=bool),
            done=np.zeros(size, dtype=bool),
            failed=np.zeros(size, dtype=bool)
        )

    def __post_init__(self):
        assert len(self) == len(self.failed)
        assert len(self) == len(self.done)
        assert len(self) == len(self.available)
        assert np.sum((~self.done) & self.failed) == 0

    def _is_selectable(self, index: Index) -> bool:
        print("SINGLE INDEX:", index)
        return (not self.done[index]) and (self.available[index]) and (not self.failed[index])

    def _is_settable(self, index: Index) -> bool:
        return (not self.done[index]) and (not self.available[index]) and (not self.failed[index])

    def select(self, index: Index) -> None:
        if not self._is_selectable(index):
            raise RuntimeError(f"Tried to select unselectable item at index '{index}'.")
        self.available[index] = False

    def set(self, index: Index, success: bool) -> None:
        if not self._is_settable(index):
            raise RuntimeError(f"Tried to set unsettable item at index '{index}'.")
        self.done[index] = True
        self.failed[index] = not success

    def reset(self, index: Index) -> None:
        self.done[index] = False
        self.failed[index] = False
        self.available[index] = True

    def list_done(self, include_failures=False) -> Collection[bool]:
        if include_failures:
            # All done, failed or not
            return self.done
        else:
            # All not failed
            return (~self.failed) & self.done

    def list_available(self) -> Collection[bool]:
        return (~self.done) & self.available

    def __len__(self) -> int:
        return len(self.done)


@dataclass(slots=True, frozen=True)
class IndexedSingleFloatTargetSurrogateProvider(ami.abc.SurrogateProviderInterface):
    features: np.ndarray = MISSING
    targets: np.ndarray = MISSING
    _schema: ami.abc.SchemaInterface = MISSING

    @classmethod
    def from_size_and_schema(cls, size: int):
        features = np.arange(size, dtype=int)
        targets = np.empty(size, dtype=float)
        schema = Schema(input_schema=[('index', int)], output_schema=[('target', float)])
        return cls(
            features=features,
            targets=targets,
            _schema=schema
        )

    def __post_init__(self):
        assert len(self.features) == len(self.targets)
        assert len(self.features) == len(self)

    def known(self, state: ami.abc.StateMachineInterface) -> Tuple[Sequence[Feature], Sequence[Target]]:
        done: Collection[bool] = state.list_done(include_failures=False)
        return self.features[done], self.targets[done]

    def unknown(self, state: ami.abc.StateMachineInterface) -> Sequence[Feature]:
        available: Collection[bool] = state.list_available()
        return self.features[available]

    def set_target(self, index: Index, value: Option[Target]) -> None:
        match value:
            case Some(v):
                self.targets[index] = v
            case Nothing:
                pass

    def __len__(self):
        return len(self.features)

    def schema(self) -> ami.abc.SchemaInterface:
        return self._schema


@dataclass(slots=True, frozen=True)
class FileStreamerTruthProvider(ami.abc.TruthProviderInterface):
    filenames: List[Path]
    _schema: ami.abc.SchemaInterface

    def parameters(self, index: Index, state: ami.abc.StateMachineInterface) -> Option[OpaqueParameters]:
        if index >= len(self):
            return Nothing
        state.select(index)
        fpath = self.filenames[index]
        data = fpath.read_bytes()
        return Some({"cif_content": data, "subdir": str(index)})

    def __len__(self) -> int:
        return len(self.filenames)

    def schema(self) -> ami.abc.SchemaInterface:
        return self._schema

    @classmethod
    def from_list_in_file(cls, path: Union[str, Path], schema: ami.abc.SchemaInterface):
        filenames = []
        with Path(path).open(mode="r") as fd:
            for line in fd:
                p = Path(line.strip())
                #FIXME: do proper logging here
                assert p.exists()
                filenames.append(p)
        return cls(filenames=filenames, _schema=schema)


@dataclass(slots=True, frozen=True)
class CsvPersistence:
    writer: IOBase

    @classmethod
    def from_filename(cls, path: Union[str, Path]):
        path = Path(path)
        writer = path.open(mode="w")
        return cls(writer)

    def __post_init__(self):
        headers = (
            "#AMI0.0.1",
        )
        for line in headers:
            print(line, file=self.writer)

    def __del__(self):
        if not self.writer.closed:
            self.writer.flush()
            self.writer.close()

    def append_valid_result(self, index: Index, value):
        print(f"{index:d},{value}", file=self.writer)
        self.writer.flush()

    def append_invalid_result(self, index: Index):
        print(f"#{index:d},", file=self.writer)


@dataclass(slots=True, frozen=True)
class InMemoryDataManager(ami.abc.DataManagerInterface):
    state: ami.abc.StateMachineInterface = MISSING
    surrogate: ami.abc.SurrogateProviderInterface = MISSING
    truth: ami.abc.TruthProviderInterface = MISSING
    io: CsvPersistence = MISSING

    @classmethod
    def from_indexed_list_in_file(cls, 
                                  path: Union[str, Path], 
                                  calc_schema: ami.abc.SchemaInterface,
                                  surrogate_schema: ami.abc.SchemaInterface,
                                  csv_filename: Union[str, Path]='AMI.out'
                                  ):
        path = Path(path)
        assert path.exists()
        truth = FileStreamerTruthProvider.from_list_in_file(path, calc_schema)
        size = len(truth)
        surrogate = IndexedSingleFloatTargetSurrogateProvider.from_size_and_schema(size)
        state = InMemoryStateMachine.from_size(size)
        io = CsvPersistence.from_filename(csv_filename)
        return cls(
            state=state,
            surrogate=surrogate,
            truth=truth,
            io=io
        )

    def available_for_calculation(self) -> Sequence[Index]:
        return np.flatnonzero(self.state.list_available())

    def set_result(self, index: Index, value: Option[Target]) -> Result[..., Exception]:
        self.surrogate.set_target(index, value)
        # FIXME: normalise to all Option or No Option, not a mixture.
        match value:
            case Some(v):
                self.state.set(index, True)
                self.io.append_valid_result(index, v)
            case Nothing:
                self.state.set(index, False)
                self.io.append_invalid_result(index)
        return Ok(())

    def unknown(self) -> Sequence[Feature]:
        return self.surrogate.unknown(self.state)

    def known(self) -> Tuple[Sequence[Feature], Sequence[Target]]:
        return self.surrogate.known(self.state)

    def __len__(self) -> int:
        return len(self.state)

    def parameters(self, index: Index) -> Option[OpaqueParameters]:
        return self.truth.parameters(index, self.state)

# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import io
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import safetensors
import torch

from mergekit.io.lazy_unpickle import (
    DeferredLoad,
    LazyTorchUnpickler,
    TorchArchiveReader,
    torch_lazy_load,
)


class TensorLoader(ABC):
    """Base class for (potentially lazy) tensor loaders."""

    @abstractmethod
    def get_tensor(self, key: str) -> torch.Tensor: ...

    @abstractmethod
    def keys(self) -> Sequence[str]: ...

    @classmethod
    def get(
        cls,
        shard_path: str,
        use_lazy_unpickle: bool = False,
        device: Optional[str] = None,
    ) -> "TensorLoader":
        if shard_path.lower().endswith(".safetensors"):
            # not a subclass of TensorLoader, but exposes same api
            return safetensors.safe_open(
                shard_path, framework="pt", device=device or "cpu"
            )
        elif use_lazy_unpickle:
            return LazyPickleLoader(shard_path, device=device)
        return DumbPytorchLoader(shard_path, device=device)


class LazyPickleLoader(TensorLoader):
    """Loader for pytorch files using a custom unpickler and vigorous monkeypatching."""

    zip_reader: TorchArchiveReader
    index: Dict[str, DeferredLoad]
    device: Optional[str] = None

    def __init__(self, path: str, device: Optional[str] = None):
        self.zip_reader = TorchArchiveReader(path)
        self.device = device
        with torch_lazy_load():
            # Read `data.pkl` out of the torch zip archive and unpickle it
            # with LazyTorchUnpickler directly. We can't go through
            # ``torch.load(pickle_module=...)`` because torch >= 2.5's
            # ``torch.serialization._load`` overrides
            # ``unpickler.persistent_load`` as an instance attribute after
            # instantiation, which silently trumps our class-level override
            # and skips DeferredLoad creation entirely. See
            # ``docs/gemma_support.md`` for the full triage.
            pkl_bytes = _read_pickle_record(self.zip_reader)
            self.index = LazyTorchUnpickler(io.BytesIO(pkl_bytes)).load()

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.index:
            raise KeyError(key)

        return self.index[key].execute(self.zip_reader, map_location=self.device)

    def keys(self) -> Sequence[str]:
        return self.index.keys()


class DumbPytorchLoader(TensorLoader):
    """Naive `torch.load` shard loading."""

    tensors: Dict[str, torch.Tensor]

    def __init__(self, path: str, device: Optional[str] = None):
        self.tensors = torch.load(path, map_location=device, weights_only=True)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def keys(self) -> Sequence[str]:
        return self.tensors.keys()


def _read_pickle_record(reader: TorchArchiveReader) -> bytes:
    """Read the ``data.pkl`` record from a torch zip archive.

    Torch archives place the pickle either under ``archive/data.pkl`` or
    ``<archive_name>/data.pkl`` depending on how ``torch.save`` was invoked.
    """
    for candidate in (
        "archive/data.pkl",
        f"{reader.archive_name}/data.pkl",
    ):
        try:
            with reader.archive.open(candidate, mode="r") as fd:
                return fd.read()
        except KeyError:
            continue
    raise RuntimeError(
        "No data.pkl record found in torch archive "
        f"{reader.archive.filename!r}; expected under 'archive/' or "
        f"'{reader.archive_name}/'."
    )

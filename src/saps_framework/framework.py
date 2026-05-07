from abc import ABC, abstractmethod


class Framework(ABC):
    # Benchmark Format -> Eager Tensor
    @abstractmethod
    def from_binsparse(self, array):
        pass

    # Eager Tensor -> Benchmark Format
    @abstractmethod
    def to_binsparse(self, array):
        pass

    # Eager Tensor -> Lazy Tensor
    @abstractmethod
    def lazy(self, array):
        pass

    # Lazy Tensor -> Eager Tensor
    @abstractmethod
    def compute(self, array):
        pass

    @abstractmethod
    def einsum(self, prgm, **kwargs):
        pass

    @abstractmethod
    def with_fill_value(self, array, value):
        pass

    @abstractmethod
    def __getattr__(self, name):
        pass

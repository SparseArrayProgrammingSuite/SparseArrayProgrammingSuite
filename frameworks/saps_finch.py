import numpy as np
import finch as ft
from binsparse import (
    COORMatrix,
    CSCMatrix,
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
    SparseLevel,
)
from binsparse.conversions import from_numpy, to_numpy, to_scipy
from finch.autoschedule import COMPILE_JULIA_GALLEY

from saps_framework import Framework


class FinchFramework(Framework):
    """SAPS adapter for Finch's Julia backend with the Galley scheduler."""

    def from_binsparse(self, array):
        match array:
            case DVECVector() | DMATRMatrix() | DMATCMatrix():
                return ft.asarray(to_numpy(array))
            case CustomTensor(shape=(), transpose=None, level=ElementLevel()):
                return ft.asarray(to_numpy(array))
            case CustomTensor(
                shape=shape,
                transpose=None,
                level=DenseLevel(rank=rank, level=ElementLevel()),
            ) if rank == len(shape):
                return ft.asarray(to_numpy(array))
            case CSCMatrix() | CSRMatrix() | COORMatrix():
                return ft.asarray(to_scipy(array).tocsr())
            case CustomTensor(
                shape=(size,),
                transpose=None,
                level=SparseLevel(
                    rank=1,
                    level=ElementLevel(values=values),
                    indices=(indices,),
                    pointers_to_next=None,
                ),
            ):
                values = np.asarray(values)
                indices = np.asarray(indices)
                fill_value = array.fill_value if array.fill is True else 0
                index_type = ft.ftype(indices.dtype)
                element_format = ft.element(
                    fill_value=fill_value,
                    element_type=ft.ftype(values.dtype),
                    position_type=index_type,
                )
                return ft.FiberTensor(
                    ft.SparseListLevel(
                        ft.ElementLevel(element_format, ft.NumpyBuffer(values)),
                        dimension=indices.dtype.type(size),
                        ptr=ft.NumpyBuffer(
                            np.array([0, len(values)], dtype=indices.dtype)
                        ),
                        idx=ft.NumpyBuffer(indices),
                    )
                )
            case CustomTensor():
                raise NotImplementedError(
                    "Finch only supports rank-1 flat sparse CustomTensor inputs."
                )
            case _:
                raise NotImplementedError(
                    f"Finch does not support BinSparse input {type(array).__name__}."
                )

    def to_binsparse(self, array):
        return from_numpy(np.asarray(array))

    def lazy(self, array):
        return ft.defer(array)

    def compute(self, array):
        return ft.compute(array, ctx=COMPILE_JULIA_GALLEY)

    def einsum(self, prgm, **kwargs):
        inputs = {name: ft.defer(value) for name, value in kwargs.items()}
        return self.compute(ft.einop(prgm, **inputs))

    def unfold(
        self,
        x,
        kernel_shape,
        *,
        axes=None,
        strides=None,
        dilations=None,
        padding=None,
        fill_value=0,
    ):
        return self.compute(
            ft.unfold(
                x,
                kernel_shape,
                axes=axes,
                strides=strides,
                dilations=dilations,
                padding=padding,
                fill_value=fill_value,
            )
        )

    def with_fill_value(self, array, value):
        if value != 0:
            raise ValueError("Finch SAPS adapter supports only a zero fill value.")
        return array

    def __getattr__(self, name):
        return getattr(ft, name)


xp = FinchFramework()

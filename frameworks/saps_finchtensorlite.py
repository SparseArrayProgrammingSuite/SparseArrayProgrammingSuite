import numpy as np
import finchlite
from saps_framework import BinsparseFormat,Framework,einsum

class FinchTensorLiteFramework(Framework):
    def __init__(self):
        pass
    def from_binsparse(self, array):
        if array.data["format"]=="dense":
            np_arr = np.array(array.data["values"]).reshape(array.data["shape"])
            return finchlite.asarray(np_arr)
        if array.data["format"]=="COO":
            indices = []
            idx_dim = 0
            while "indices_" + str(idx_dim) in array.data:
                indices.append(array.data["indices_" + str(idx_dim)])
                idx_dim += 1
            values = array.data["values"]
            shape = array.data["shape"]
            dense = np.zeros(shape, dtype=values.dtype)
            dense[tuple(indices)] = values
            return finchlite.asarray(dense)
        raise ValueError("Unsupported format: " + array.data["format"])
    
    def to_binsparse(self, array):
        return BinsparseFormat.from_numpy(np.asarray(array))
    
    def lazy(self, array):
        return finchlite.lazy(array)
    
    def compute(self, array):
        return finchlite.compute(array)
    
    def einsum(self, prgm, **kwargs):
        return einsum(self,prgm,**kwargs)
    
    def with_fill_value(self, array, value):
        return array
    
    def __getattr__(self, name):
        return getattr(finchlite,name)
    
xp = FinchTensorLiteFramework()
        




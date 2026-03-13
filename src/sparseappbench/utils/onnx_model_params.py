import numpy as np
import onnx
from onnx import TensorProto

def _tensor_to_numpy(tensor):

    if tensor.data_type == TensorProto.FLOAT:
        dtype = np.float32
    elif tensor.data_type == TensorProto.INT64:
        dtype = np.int64
    else:
        raise NotImplementedError

    if tensor.raw_data:
        arr = np.frombuffer(tensor.raw_data, dtype=dtype).copy()
    elif tensor.data_type == TensorProto.FLOAT:
        arr = np.array(tensor.float_data, dtype=dtype)
    else:
        arr = np.array(tensor.int64_data, dtype=dtype)

    return arr.reshape(tuple(tensor.dims))


def load_initializers(onnx_path):
    model = onnx.load(onnx_path)
    params = {}
    for init in model.graph.initializer:
        params[init.name] = _tensor_to_numpy(init)
    return params

def _find(params, names):
    for name in names:
        if name in params:
            return params[name]
    for name in names:
        for key in params:
            if key.endswith(name):
                return params[key]
    raise KeyError(names)


def load_conv2_params(onnx_path):
    params = load_initializers(onnx_path)
    return {
        "conv1_w": _find(params, ["conv1.weight", "layers.0.conv1.weight"]),
        "conv1_b": _find(params, ["conv1.bias", "layers.0.conv1.bias"]),
        "conv2_w": _find(params, ["conv2.weight", "layers.0.conv2.weight"]),
        "conv2_b": _find(params, ["conv2.bias", "layers.0.conv2.bias"]),
        "fc1_w": _find(params, ["fc1.weight"]),
        "fc1_b": _find(params, ["fc1.bias"]),
        "fc2_w": _find(params, ["fc2.weight"]),
        "fc2_b": _find(params, ["fc2.bias"]),
        "fc3_w": _find(params, ["fc3.weight"]),
        "fc3_b": _find(params, ["fc3.bias"]),
    }

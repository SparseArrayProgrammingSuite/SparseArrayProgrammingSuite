# onnxpy

Compile ONNX models to inline Python functions backed by an Array API-compatible
namespace. Each ONNX node becomes statements in one function. The compiler
returns only that function's source and an ordered dictionary of tensor inputs;
it emits no imports, module globals, helper functions, or application wrappers.

## Install

Requires Python 3.11+ and Poetry.

```bash
poetry install
```

## Usage

```python
import math

import numpy as np
import onnx
from onnxpy import compile_to_source

model = onnx.load("model.onnx")
source, tensors = compile_to_source(model)
# source starts with: def model(xp, input, weight, ...):
scope = {"math": math}
exec(source, scope)

# Pass graph inputs (excluding initializers), then tensors in dictionary order.
result = scope["model"](np, input_array, *tensors.values())
```

`xp` is an explicit first parameter. Every input has a named positional
parameter. The tensor dictionary contains initializers followed by tensor
`Constant` nodes. Multiple graph outputs are returned as a tuple.

The caller provides `math` in the function's globals for models using `Flatten`.
The caller also owns compiling the source or saving it to a file.

For embedding in another application, set the function name and any additional
parameters that application passes before the tensor inputs:

```python
source, tensors = compile_to_source(
    model, function_name="inference", extra_parameters=("context",)
)
# def inference(xp, context, input, weight, ...):
```

Extra parameters are unused by the computation. Their names are reserved during
emission, so a graph value named `context` is renamed without changing the
caller's parameter. Names must be unique Python identifiers and cannot shadow
Python builtins, `xp`, `np`, `math`, or compiler temporaries (`_onnxpy_*`).

## Convolution and pooling

Models with convolution or pooling require an array namespace that provides
`conv` and `pool` in addition to Array API operations:

```python
xp.conv(x, w, *, strides=None, dilations=None, padding=None, group=1)
xp.pool(x, kernel_shape, *, reduce, strides=None, dilations=None,
        padding=None, fill_value=0)
```

`x` has shape `(N, C, *spatial)`, `w` has shape `(M, C / group, *kernel)`, and
`padding` contains a `(before, after)` pair for each spatial axis, padded with
zeros for `conv` and with `fill_value` for `pool`. `reduce` is `"max"`, `"+"`
or `"mean"`. Plain NumPy provides neither; SAPS frameworks provide both.

Padding arithmetic is expanded at compile time, so generated Conv and pooling
code contains no Python loops. AveragePool without `count_include_pad`
divides a sum pool of the input by a sum pool of ones.
Spatial rank must be known from tensor metadata or operator attributes;
spatial dimensions may vary at runtime.

## Supported ops

`Add, AveragePool, Cast, Concat, Constant, Conv, Div, Flatten, Gemm,
GlobalAveragePool, Identity, LogSoftmax, MatMul, MaxPool, Mul, Relu,
Reshape, Sigmoid, Softmax, Sub, Tanh, Transpose`.

Use `onnxpy.supported_ops()` to inspect this set. Unsupported op types raise
`NotImplementedError` at compile time.

## Tests

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

`tests/test_ops.py` compares emitted functions against `onnxruntime` on tiny
graphs. `tests/test_emit.py` checks the function-only output and calling
convention. `tests/test_mnist.py` checks `tmp/models/mnist-8.onnx` against
`onnxruntime` on five seeds within `atol=rtol=1e-4`.

## Limitations

- No control flow (`If`, `Loop`, `Scan`).
- No sequence or optional types.
- No multi-output ops.
- Conv and pooling require the target namespace's `xp.conv` and `xp.pool` extensions.
- Modern op semantics only; pre-opset-13 quirks are not normalized.

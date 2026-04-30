import numpy as np
def forward_euler(
    xp,
    dydx,
    span,
    y0,
    first_step,
):
    """Forward Euler method of approximating ordinary differential equations (ODEs)."""
    # Builtin range function does not support floating-point step
    curr = span[0]
    inputs = []
    while curr < span[1]:
        inputs.append(curr)
        curr += first_step

    step = first_step
    outputs = [None for _ in inputs]
    outputs[0] = xp.array(y0, dtype=float).flatten()

    for i in range(1, len(inputs)):
        # y_new = y + dy/dx * delta x

        dydt_vector = xp.array(dydx(inputs[i - 1], outputs[i - 1]), dtype=np.complex128).flatten()
        outputs[i] = [outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))]

    return (inputs, outputs)


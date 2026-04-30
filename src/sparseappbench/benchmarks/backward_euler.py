def backward_euler(
    xp,
    dydx,
    span,
    y0,
    first_step,
):
    """Backward Euler method of approximating ordinary differential equations (ODEs)."""
    # Builtin range function does not support floating-point step
    curr = span[0]
    inputs = []
    while curr < span[1]:
        inputs.append(curr)
        curr += first_step

    step = first_step
    outputs = [None for _ in inputs]
    outputs[0] = y0
    # y_n+1 = y_n + dy/dx(x_n+1, y_n+1) * delta x

    # Fixed point iteration
    for i in range(1, len(inputs)):
        y_guess = outputs[i - 1] # initial guess
        for _ in range(10):
            dydt_vector = dydx(inputs[i], y_guess)
            y_guess = [outputs[i - 1][j] + dydt_vector[j] * step for j in range(len(y0))]
        outputs[i] = y_guess
    
    return (inputs, outputs)

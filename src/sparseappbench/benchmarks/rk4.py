def rk4(
    xp,
    dydx,
    span,
    y0,
    first_step
):
    """Runge-Kutta 4th order method of approximating ordinary differential equations (ODEs)."""
    curr = span[0]
    inputs = []
    while curr < span[1]:
        inputs.append(curr)
        curr += first_step

    step = first_step
    outputs = [None for _ in inputs]
    outputs[0] = y0

    for i in range(1, len(inputs)):
        y_prev = outputs[i - 1]
        k1 = dydx(inputs[i - 1], y_prev)
        k2_state = [y_prev[j] + (step / 2) * k1[j] for j in range(len(y0))]
        k2 = dydx(inputs[i - 1] + step / 2, k2_state)
        k3_state = [y_prev[j] + (step / 2) * k2[j] for j in range(len(y0))]
        k3 = dydx(inputs[i - 1] + step / 2, k3_state)
        k4_state = [y_prev[j] + step * k3[j] for j in range(len(y0))]
        k4 = dydx(inputs[i - 1] + step, k4_state)
        outputs[i] = [
            y_prev[j] + (step / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
            for j in range(len(y0))
        ]

    return (inputs, outputs)

from MLBasics.regression_core import extend_vals, least_squares_cost, least_squares_next_params


def linear_derivs(*coeffs):
    def generated_deriv(n):
        def n_deriv(*vals):
            return extend_vals(*vals)[n]
        return n_deriv
    return tuple(generated_deriv(n) for n in xrange(len(coeffs)))


def linear_function(*coeffs):
    def generated(*vals):
        vals = extend_vals(*vals)
        if len(vals) != len(coeffs):
            raise IndexError
        y = 0
        for index in range(len(coeffs)):
            y += coeffs[index] * vals[index]
        return y
    return {"func": generated, "params": coeffs, "derivs": linear_derivs(*coeffs)}


def ols_regression(lin_func, iterations, learning_rate, *pts):
    for itcount in xrange(iterations):
        nextparams = least_squares_next_params(lin_func, learning_rate, *pts)
        lin_func = linear_function(*tuple(nextparams))
        curr_cost = least_squares_cost(lin_func, *pts)
        print("Cost after iteration {} is : {}".format(itcount, curr_cost))
    return lin_func


def ols_regression_bounded(lin_func, learning_rate, bound, stop_threshold, *pts):
    curr_cost = least_squares_cost(lin_func, *pts)
    print("Initial cost is: {}".format(curr_cost))
    increasing_count = 0
    consecutive_small_count = 0
    itcount = 0
    while True:
        itcount += 1
        nextparams = least_squares_next_params(lin_func, learning_rate, *pts)
        lin_func = linear_function(*nextparams)
        last_cost, curr_cost = curr_cost, least_squares_cost(lin_func, *pts)

        diff = last_cost - curr_cost
        print("Cost after iteration {} is : {} with difference {}".format(itcount, curr_cost, diff))
        if 0 < diff < bound:
            consecutive_small_count += 1
        elif diff < 0:
            increasing_count += 1

        if consecutive_small_count > stop_threshold:
            break
        elif increasing_count > stop_threshold:
            raise OverflowError
    return lin_func


pts = (((0,), 3.25), ((1,), 5.212), ((2,), 9.8))
params = (-20, 10)

lin_func = linear_function(*params)
print("Cost 1: {}".format(least_squares_cost(lin_func, *pts)))

final_func = ols_regression_bounded(lin_func, 0.01, 0.00001, 10, *pts)

print(list(final_func["params"]))




import math

from MLBasics.regression_core import extend_vals, least_squares_cost, least_squares_next_params, dot


def logistic_derivs(logistic_func, *coeffs):
    def generated_deriv(n):
        def n_deriv(*vals):
            evals = extend_vals(*vals)
            if type(vals) != type(coeffs) or len(evals) != len(coeffs):
                print("Error in list sizes. Vals {} // Coeffs {}".format(evals, coeffs))
                raise IndexError
            prod = dot(evals, coeffs)
            deriv = -logistic_func(*vals)**2 * (-evals[n] * math.exp(-prod))
            return deriv
        return n_deriv
    return tuple(generated_deriv(n) for n in xrange(len(coeffs)))


def logistic_function(*coeffs):
    def generated(*vals):
        vals = extend_vals(*vals)
        if len(vals) != len(coeffs):
            raise IndexError
        prod = dot(vals, coeffs)
        return 1/(1 + math.exp(-prod))
    return {"func": generated, "params": coeffs, "derivs": logistic_derivs(generated, *coeffs)}

def logistic_regression(log_func, iterations, learning_rate, *pts):
    for itcount in xrange(iterations):
        nextparams = least_squares_next_params(log_func, learning_rate, *pts)
        log_func = logistic_function(*tuple(nextparams))
        curr_cost = least_squares_cost(log_func, *pts)
        print("Cost after iteration {} is : {}".format(itcount, curr_cost))
    return log_func

def logistic_regression_bounded(log_func, learning_rate, bound, stop_threshold, *pts):
    curr_cost = least_squares_cost(log_func, *pts)
    print("Initial cost is: {}".format(curr_cost))
    increasing_count = 0
    consecutive_small_count = 0
    itcount = 0
    while True:
        itcount += 1
        nextparams = least_squares_next_params(log_func, learning_rate, *pts)
        log_func = logistic_function(*nextparams)
        last_cost, curr_cost = curr_cost, least_squares_cost(log_func, *pts)

        diff = last_cost - curr_cost
        print("Cost after iteration {} is : {} with difference {}".format(itcount, curr_cost, diff))
        print("Params at iteration {} are: {}".format(itcount, log_func["params"]))
        if 0 < diff < bound:
            consecutive_small_count += 1
        elif diff < 0:
            increasing_count += 1

        if consecutive_small_count > stop_threshold:
            break
        elif increasing_count > stop_threshold:
            raise OverflowError
    return log_func


pts = (((0.1,), 0), ((0.4,), 0), ((0.3,), 0), ((0.2,), 0), ((0.75,), 1), ((0.7,), 1), ((0.9,), 1), ((0.6,), 1))
params = (-0.1, 0.2)

log_func = logistic_function(*params)
print("Cost 1: {}".format(least_squares_cost(log_func, *pts)))

final_func = logistic_regression_bounded(log_func, 0.1, 0.000001, 10, *pts)


print(list(final_func["params"]))

print("Output values: {}".format([((x, y), final_func["func"](*x)) for x, y in pts]))

print("Vals {} // Coeffs{} // Dot: {}".format(extend_vals(*pts[0][0]), final_func["params"], dot(extend_vals(*pts[0][0]), final_func["params"])))

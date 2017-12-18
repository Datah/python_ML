from regression_core import extend_vals, non_regularized_cost, next_params, non_regularized_gradient_descent_value,\
    least_squares_fcost, regularized_gradient_descent_value, squared_coefficient_regularization, \
    squared_coefficient_regularization_gradient, regularized_cost


def linear_derivs(coeffs):
    def generated_deriv(n):
        def n_deriv(vals):
            return extend_vals(vals)[n]
        return n_deriv
    return tuple(generated_deriv(n) for n in xrange(len(coeffs)))


def linear_function(coeffs):
    def generated(vals):
        vals = extend_vals(vals)
        if len(vals) != len(coeffs):
            raise IndexError
        y = 0
        for index in range(len(coeffs)):
            y += coeffs[index] * vals[index]
        return y
    funcmap = {"func": generated, "params": coeffs, "derivs": linear_derivs(coeffs)}
    costmap = least_squares_fcost(funcmap)
    funcmap.update(costmap)
    return funcmap


def ridge_regression(lin_func, iterations, learning_rate, pts, l):
    for itcount in xrange(iterations):
        gradvals = regularized_gradient_descent_value(lin_func, pts, squared_coefficient_regularization_gradient, l)
        nextparams = next_params(lin_func, learning_rate, gradvals)
        lin_func = linear_function(tuple(nextparams))
        curr_cost = regularized_cost(lin_func, pts, squared_coefficient_regularization, l)
        #print("Cost after iteration {} is : {}".format(itcount, curr_cost))
    return lin_func


def ridge_regression_bounded(lin_func, learning_rate, bound, stop_threshold, pts, l):
    curr_cost = regularized_cost(lin_func, pts, squared_coefficient_regularization, l)
    print("Initial cost is: {}".format(curr_cost))
    increasing_count = 0
    consecutive_small_count = 0
    itcount = 0
    while True:
        itcount += 1
        gradvals = regularized_gradient_descent_value(lin_func, pts, squared_coefficient_regularization_gradient, l)
        nextparams = next_params(lin_func, learning_rate, gradvals)
        lin_func = linear_function(nextparams)
        last_cost, curr_cost = curr_cost, regularized_cost(lin_func, pts, squared_coefficient_regularization, l)

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


def ols_regression(lin_func, iterations, learning_rate, pts):
    for itcount in xrange(iterations):
        gradvals = non_regularized_gradient_descent_value(lin_func, pts)
        nextparams = next_params(lin_func, learning_rate, gradvals)
        lin_func = linear_function(tuple(nextparams))
        curr_cost = non_regularized_cost(lin_func, pts)
        print("Cost after iteration {} is : {}".format(itcount, curr_cost))
    return lin_func


def ols_regression_bounded(lin_func, learning_rate, bound, stop_threshold, pts):
    curr_cost = non_regularized_cost(lin_func, pts)
    print("Initial cost is: {}".format(curr_cost))
    increasing_count = 0
    consecutive_small_count = 0
    itcount = 0
    while True:
        itcount += 1
        gradvals = non_regularized_gradient_descent_value(lin_func, pts)
        nextparams = next_params(lin_func, learning_rate, gradvals)
        lin_func = linear_function(nextparams)
        last_cost, curr_cost = curr_cost, non_regularized_cost(lin_func, pts)

        diff = last_cost - curr_cost
        #print("Cost after iteration {} is : {} with difference {}".format(itcount, curr_cost, diff))
        if 0 < diff < bound:
            consecutive_small_count += 1
        elif diff < 0:
            increasing_count += 1

        if consecutive_small_count > stop_threshold:
            break
        elif increasing_count > stop_threshold:
            raise OverflowError
    return lin_func


pts = (((0,), 3), ((1,), 6), ((2,), 9))
params = (-20, 10)

lin_func = linear_function(params)
print("Cost 1: {}".format(non_regularized_cost(lin_func, pts)))

final_func = ridge_regression_bounded(lin_func, 0.1, 0.000001, 10, pts, 0.01)

print(list(final_func["params"]))

final_func = ols_regression_bounded(lin_func, 0.1, 0.000001, 10, pts)

print(list(final_func["params"]))




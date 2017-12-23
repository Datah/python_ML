from data_utils import standardize
from function_definitions import logistic_function
from regression_core import extend_vals, non_regularized_cost, next_params, dot, \
    non_regularized_gradient_descent_value, regularized_gradient_descent_value, regularized_cost, \
    squared_coefficient_regularization, squared_coefficient_regularization_gradient


def logistic_regression(log_func, iterations, learning_rate, pts):
    for itcount in range(iterations):
        gradvals = non_regularized_gradient_descent_value(log_func, pts)
        nextparams = next_params(log_func, learning_rate, gradvals)
        log_func = logistic_function(tuple(nextparams))
        curr_cost = non_regularized_cost(log_func, pts)
        print("Cost after iteration {} is : {}".format(itcount, curr_cost))
    return log_func


def logistic_regression_bounded(log_func, learning_rate, bound, stop_threshold, pts):
    curr_cost = non_regularized_cost(log_func, pts)
    print("Initial cost is: {}".format(curr_cost))
    increasing_count = 0
    consecutive_small_count = 0
    itcount = 0
    while True:
        itcount += 1
        gradvals = non_regularized_gradient_descent_value(log_func, pts)
        nextparams = next_params(log_func, learning_rate, gradvals)
        log_func = logistic_function(nextparams)
        last_cost, curr_cost = curr_cost, non_regularized_cost(log_func, pts)

        diff = last_cost - curr_cost
        #print("Cost after iteration {} is : {} with difference {}".format(itcount, curr_cost, diff))
        #print("Params at iteration {} are: {}".format(itcount, log_func["params"]))
        if 0 < diff < bound:
            consecutive_small_count += 1
        elif diff < 0:
            increasing_count += 1

        if consecutive_small_count > stop_threshold:
            break
        elif increasing_count > stop_threshold:
            raise OverflowError
    return log_func


def regularized_logistic_regression(log_func, iterations, learning_rate, pts, l):
    for itcount in range(iterations):
        gradvals = regularized_gradient_descent_value(log_func, pts, squared_coefficient_regularization_gradient, l)
        nextparams = next_params(log_func, learning_rate, gradvals)
        log_func = logistic_function(tuple(nextparams))
        curr_cost = regularized_cost(log_func, pts, squared_coefficient_regularization, l)
        print("Cost after iteration {} is : {}".format(itcount, curr_cost))
    return log_func


def regularized_logistic_regression_bounded(log_func, learning_rate, bound, stop_threshold, pts, l):
    curr_cost = regularized_cost(log_func, pts, squared_coefficient_regularization, l)
    print("Initial cost is: {}".format(curr_cost))
    increasing_count = 0
    consecutive_small_count = 0
    itcount = 0
    while True:
        itcount += 1
        gradvals = regularized_gradient_descent_value(log_func, pts, squared_coefficient_regularization_gradient, l)
        nextparams = next_params(log_func, learning_rate, gradvals)
        log_func = logistic_function(nextparams)
        last_cost, curr_cost = curr_cost, regularized_cost(log_func, pts, squared_coefficient_regularization, l)

        diff = last_cost - curr_cost
        # print("Cost after iteration {} is : {} with difference {}".format(itcount, curr_cost, diff))
        # print("Params at iteration {} are: {}".format(itcount, log_func["params"]))
        if 0 < diff < bound:
            consecutive_small_count += 1
        elif diff < 0:
            increasing_count += 1

        if consecutive_small_count > stop_threshold:
            break
        elif increasing_count > stop_threshold:
            raise OverflowError
    return log_func


pts = (((0.1, 0.8, 0.2), 0), ((0.4, 0.92, 0.35), 0), ((0.3, 0.89, 0.22), 0), ((0.2, 0.81, 0.27), 0),
       ((0.75, 0.4, 0.4), 1), ((0.7, 0.79, 0.4), 1), ((0.9, 0.52, 0.73), 1), ((0.6, 0.01, 0.99), 1))

stdpts_with_stats = standardize(pts)
stdpts = stdpts_with_stats["points"]
stats = stdpts_with_stats["stats"]
print("Standardized points: {} // stats {}".format(stdpts, stats))
params = (0.1, 0.1, 0.1, 0.1)

log_func = logistic_function(params)
print("Cost 1: {}".format(non_regularized_cost(log_func, stdpts)))

final_func = regularized_logistic_regression_bounded(log_func, 0.1, 0.000001, 10, stdpts, 0.01)


print(list(final_func["params"]))

print("Output values: {}".format([((x, y), final_func["func"](x)) for x, y in stdpts]))

print("Vals {} // Coeffs{} // Dot: {}".format(extend_vals(pts[0][0]), final_func["params"], dot(extend_vals(pts[0][0]), final_func["params"])))

final_func = logistic_regression_bounded(log_func, 0.1, 0.000001, 10, stdpts)


print(list(final_func["params"]))

print("Output values: {}".format([((x, y), final_func["func"](x)) for x, y in stdpts]))

print("Vals {} // Coeffs{} // Dot: {}".format(extend_vals(pts[0][0]), final_func["params"], dot(extend_vals(pts[0][0]), final_func["params"])))

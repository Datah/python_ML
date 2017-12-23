from data_utils import standardize
from function_definitions import linear_function
from regression_core import non_regularized_cost, next_params, non_regularized_gradient_descent_value, \
    regularized_gradient_descent_value, squared_coefficient_regularization, \
    squared_coefficient_regularization_gradient, regularized_cost


def ridge_regression(lin_func, iterations, learning_rate, pts, l):
    for itcount in range(iterations):
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
    for itcount in range(iterations):
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
stdpts_with_stats = standardize(pts)
stdpts = stdpts_with_stats["points"]
stats = stdpts_with_stats["stats"]
print("Standardized points: {}".format(stdpts))
params = (-20, 10)

lin_func = linear_function(params)
print("Cost 1: {}".format(non_regularized_cost(lin_func, stdpts)))

final_func = ridge_regression_bounded(lin_func, 0.01, 0.000001, 10, stdpts, 200)

print(list(final_func["params"]))

final_func = ols_regression_bounded(lin_func, 0.1, 0.000001, 10, stdpts)

print(list(final_func["params"]))




import math


def extend_vals(vals):
    return (1,) + vals


def least_squares_fcost(func_premap):
    func = func_premap["func"]

    def cost(x, y):
        cost = 0.5 * (func(x) - y)**2
        return cost

    def deriv(n, x, y):
        derivs = func_premap["derivs"]
        cost_deriv = (func(x) - y) * derivs[n](x)
        return cost_deriv
    return {"cost": cost, "cost_deriv": deriv}


def logistic_fcost(func_premap):
    func = func_premap["func"]

    def cost(x, y):
        cost = -math.log(func(x)) if y == 1 else -math.log(1-func(x))
        return cost

    def deriv(n, x, y):
        derivs = func_premap["derivs"]
        cost_deriv = -1/math.fabs(func(x)) * derivs[n](x) if y == 1 else 1/math.fabs(1-func(x)) * derivs[n](x)
        return cost_deriv
    return {"cost": cost, "cost_deriv": deriv}


def squared_coefficient_regularization(params):
    return sum(params[n]**2 for n in range(1, len(params)))


def squared_coefficient_regularization_gradient(params):
    pre = tuple(2*params[n] for n in range(1, len(params)))
    return (0,) + pre

def no_regularization(params):
    return 0

def no_regularization_gradient(params):
    return tuple(0 for n in range(len(params)))


def non_regularized_cost(funcmap, pts):
    np = len(pts)
    cost_function = funcmap["cost"]
    cost_sum = math.fsum(cost_function(x, y) for x, y in pts)
    return cost_sum / np


def non_regularized_gradient_descent_value(funcmap, pts):
    derivs = funcmap["derivs"]
    cost_function_derivative = funcmap["cost_deriv"]
    numcoeffs = len(derivs)
    np = len(pts)
    gradvals = tuple(math.fsum(cost_function_derivative(n, x, y) for x, y in pts) / np for n in range(numcoeffs))
    return gradvals


def regularized_cost(funcmap, pts, regularization_function = no_regularization, l = 0):
    params = funcmap["params"]
    np = len(pts)
    cost_sum = non_regularized_cost(funcmap, pts) + l / np * regularization_function(params)
    return cost_sum


def regularized_gradient_descent_value(funcmap, pts, regularization_gradient = no_regularization_gradient, l = 0):
    rgrad = regularization_gradient(funcmap["params"])
    gradvals = non_regularized_gradient_descent_value(funcmap, pts)
    np = len(pts)
    gradvals_final = tuple(gradvals[n] + l / np * rgrad[n] for n in range(len(gradvals)))
    return gradvals_final


def dot(a, b):
    if not (type(a) == tuple or type(a) == list) or not (type(b) == tuple or type(b) == list):
        print("Error doing dot product for {} with {}".format(a, b))
        raise TypeError
    return sum(a[n] * b[n] for n in range(0, len(a)))


def mat_mult(m, n):
    mat = []
    for j in range(len(m)):
        row = m[j]
        ncols = len(n[0])
        cols = [[r[c] for r in n] for c in range(ncols)]
        prn = [dot(row, cols[c]) for c in range(ncols)]
        mat.append(tuple(prn))
    return tuple(mat)


def init_matrix(n, m, func):
    return tuple(tuple(func(i, j) for i in range(n)) for j in range(m))


def val_matrix(n, m, val = 0):
    return init_matrix(n, m, const_func(val))


def const_func(val):
    def cfunc(i, j):
        return val
    return cfunc


def next_params(funcmap, learning_rate, gradvals):
    params = funcmap["params"]
    newcoeffs = tuple(params[n] - learning_rate * gradvals[n] for n in range(len(params)))
    return newcoeffs

mat_one = ((0, 1), (1, 0))
mat_two = ((3,), (1,))
print("{}".format(mat_mult(mat_one, mat_mult(mat_one, mat_two))))
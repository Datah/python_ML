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


def cost(funcmap, pts):
    np = len(pts)
    cost_function = funcmap["cost"]
    cost_sum = math.fsum(cost_function(x, y) for x, y in pts)
    return cost_sum / np


def dot(a, b):
    if not (type(a) == tuple or type(a) == list) or not (type(b) == tuple or type(b) == list):
        print("Error doing dot product for {} with {}".format(a, b))
        raise TypeError
    return sum(a[n] * b[n] for n in range(0, len(a)))


def gradient_descent_value(funcmap, pts):
    derivs = funcmap["derivs"]
    cost_function_derivative = funcmap["cost_deriv"]
    numcoeffs = len(derivs)
    np = len(pts)
    gradvals = tuple(math.fsum(cost_function_derivative(n, x, y) for x, y in pts) / np for n in xrange(numcoeffs))
    return gradvals


def next_params(funcmap, learning_rate, gradvals):
    params = funcmap["params"]
    newcoeffs = tuple(params[n] - learning_rate * gradvals[n] for n in xrange(len(params)))
    return newcoeffs
import math


def extend_vals(*vals):
    return (1,) + vals


def least_squares_cost(funcmap, *pts):
    func = funcmap["func"]
    np = len(pts)
    cost_sum = math.fsum((y - func(*x))**2 for x, y in pts)
    return cost_sum / (2 * np)


def dot(a, b):
    if not (type(a) == tuple or type(a) == list) or not (type(b) == tuple or type(b) == list):
        print("Error doing dot product for {} with {}".format(a, b))
        raise TypeError
    return sum(a[n] * b[n] for n in range(0, len(a)))


def least_squares_gradient_descent_value(funcmap, *pts):
    func = funcmap["func"]
    derivs = funcmap["derivs"]
    numcoeffs = len(derivs)
    np = len(pts)
    #print("Points: {}".format(pts))
    #for n in xrange(numcoeffs):
    #    print("Derivative {} at input {}: ".format(n, pts[0][0]))
    #    print("{}".format(derivs[n](*(pts[0][0]))))
    gradvals = tuple(math.fsum(-(y - func(*x))*derivs[n](*x) for x, y in pts) / np for n in xrange(numcoeffs))
    return gradvals


def least_squares_next_params(funcmap, learning_rate, *pts):
    params = funcmap["params"]
    grdv = least_squares_gradient_descent_value(funcmap, *pts)
    newcoeffs = tuple(params[n] - learning_rate * grdv[n] for n in xrange(len(params)))
    return newcoeffs